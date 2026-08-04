#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
using namespace sycl;

int main() {
    const int H=2048, I=512;
    printf("Finding B70...\n"); fflush(stdout);
    
    device b70;
    bool found=false;
    for (auto &p : platform::get_platforms()) {
        for (auto &d : p.get_devices()) {
            if (!d.is_gpu()) continue;
            auto vid = d.get_info<info::device::vendor_id>();
            auto mem = d.get_info<info::device::global_mem_size>();
            if (vid==0x8086 && mem>8e9 && mem<40e9) { b70=d; found=true; break; }
        }
        if (found) break;
    }
    if (!found) { fprintf(stderr,"B70 not found\n"); return 1; }

    queue q(b70, property_list{property::queue::enable_profiling()});
    printf("Queue: %s\n\n", q.get_device().get_info<info::device::name>().c_str());

    // Device buffers
    float *d_x = malloc_device<float>(H, q);
    float *d_w = malloc_device<float>(I*H, q);  // weight [I, H]
    float *d_inter = malloc_device<float>(I, q);
    float *d_out = malloc_device<float>(H, q);

    // Host buffers — properly sized
    float *h_x = (float*)malloc(H*sizeof(float));
    float *h_w = (float*)malloc(I*H*sizeof(float));
    float *h_out = (float*)malloc(H*sizeof(float));
    for (int i=0;i<H;i++) h_x[i]=0.1f;
    for (int i=0;i<I*H;i++) h_w[i]=0.01f;

    // Upload
    q.memcpy(d_x, h_x, H*sizeof(float)).wait();
    q.memcpy(d_w, h_w, I*H*sizeof(float)).wait();
    printf("Upload OK\n");

    // Warmup
    for (int i=0;i<10;i++)
        q.submit([&](handler &h){
            h.parallel_for(I,[=](auto j){ float a=0; for(int k=0;k<H;k++) a+=d_x[k]*d_w[j*H+k]; d_inter[j]=a; });
        }).wait();
    printf("Warmup OK\n\n");

    // Profile simple GEMV: each WI computes one row of gate output
    printf("=== Kernel profiling ===\n");
    auto ev = q.submit([&](handler &h){
        h.parallel_for(I,[=](auto j){ float a=0; for(int k=0;k<H;k++) a+=d_x[k]*d_w[j*H+k]; d_inter[j]=a; });
    });
    ev.wait();
    printf("Simple GEMV (parallel_for I=512): %.1f us\n",
        (ev.get_profiling_info<info::event_profiling::command_end>() - 
         ev.get_profiling_info<info::event_profiling::command_start>())/1e3);

    // nd_range with different wg sizes
    for (int wgs : {128, 256, 512}) {
        auto e = q.submit([&](handler &h){
            h.parallel_for(nd_range<1>(wgs,wgs),[=](nd_item<1> it){
                int lid=it.get_local_id(0);
                for(int j=lid;j<I;j+=wgs){ float a=0; for(int k=0;k<H;k++) a+=d_x[k]*d_w[j*H+k]; d_inter[j]=a; }
            });
        });
        e.wait();
        printf("nd_range GEMV (wg=%d): %.1f us\n", wgs,
            (e.get_profiling_info<info::event_profiling::command_end>() - 
             e.get_profiling_info<info::event_profiling::command_start>())/1e3);
    }

    // Full fused expert (gate+up+SiLU+down) in one kernel
    float *d_wd = malloc_device<float>(H*I, q); // down weight [H, I]
    q.memcpy(d_wd, h_w, H*I*sizeof(float)).wait();
    
    printf("\nFull expert kernel (gate+up+SiLU+down, wg=512):\n");
    for (int trial=0; trial<3; trial++) {
        auto e = q.submit([&](handler &h){
            h.parallel_for(nd_range<1>(512,512),[=](nd_item<1> it){
                int lid=it.get_local_id(0);
                // Phase 1: gate+up GEMV
                for(int j=lid;j<I;j+=512){
                    float g=0;
                    for(int k=0;k<H;k++) g+=d_x[k]*d_w[j*H+k];
                    d_inter[j]=g/(1.0f+(float)sycl::exp(-g))*g; // silu approx
                }
                group_barrier(it.get_group());
                // Phase 2: down GEMV
                for(int j=lid;j<H;j+=512){
                    float a=0;
                    for(int k=0;k<I;k++) a+=d_inter[k]*d_wd[j*I+k];
                    d_out[j]=a;
                }
            });
        });
        e.wait();
        printf("  trial %d: %.1f us\n", trial,
            (e.get_profiling_info<info::event_profiling::command_end>() - 
             e.get_profiling_info<info::event_profiling::command_start>())/1e3);
    }

    // Full dispatch wall clock: H2D + zero + expert + D2H
    printf("\n=== Full dispatch wall clock ===\n");
    auto t0=std::chrono::high_resolution_clock::now();
    for(int iter=0;iter<100;iter++){
        q.memcpy(d_x,h_x,H*sizeof(float)).wait();
        q.submit([&](handler&h){h.parallel_for(H,[=](auto i){d_out[i]=0;});}).wait();
        q.submit([&](handler&h){
            h.parallel_for(nd_range<1>(512,512),[=](nd_item<1> it){
                int lid=it.get_local_id(0);
                for(int j=lid;j<I;j+=512){float g=0;for(int k=0;k<H;k++)g+=d_x[k]*d_w[j*H+k];d_inter[j]=g;}
                group_barrier(it.get_group());
                for(int j=lid;j<H;j+=512){float a=0;for(int k=0;k<I;k++)a+=d_inter[k]*d_wd[j*I+k];d_out[j]=a;}
            });
        }).wait();
        q.memcpy(h_out,d_out,H*sizeof(float)).wait();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/100;
    printf("Per dispatch: %.1f us\n",us);
    printf("3 experts x 40 layers: %.2f ms\n", us*3*40/1e3);

    // Just the H2D+D2H overhead (no kernel)
    auto t2=std::chrono::high_resolution_clock::now();
    for(int iter=0;iter<100;iter++){
        q.memcpy(d_x,h_x,H*sizeof(float)).wait();
        q.memcpy(h_out,d_out,H*sizeof(float)).wait();
    }
    auto t3=std::chrono::high_resolution_clock::now();
    printf("\nH2D+D2H only: %.1f us\n", std::chrono::duration<double,std::micro>(t3-t2).count()/100);

    free(d_x,q);free(d_w,q);free(d_inter,q);free(d_out,q);free(d_wd,q);
    free(h_x);free(h_w);free(h_out);
    printf("\n[done]\n");
}
