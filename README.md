# Shooting Brake

![Shooting Brake](assets/Indecent-Porsche-911-Shooting-Brake-1.webp)

> A shooting brake was never for everyone - it's the rare machine that refuses to
> sacrifice speed for capacity, built in limited numbers for people who wanted both.

One 118B MoE model — NVFP4, 58.2 GiB of weights, 131,072 context — split across two GPU
vendors inside every forward pass. An **RTX 5090** runs attention, the router, and the
local and shared experts. **Two Intel Arc Pro B70s** hold and compute 170 routed experts,
85 per card, reached over a pinned-host-memory doorbell: 47 host-visible sync points per
token, no shared runtime between the vendors.

```bash
./serve_production.sh          # OpenAI-compatible server on :8017
```

## Against one RTX PRO 6000 (96 GB), same checkpoint, same harness

| | Shooting Brake (~$9k) | PRO 6000 (~$16k) |
|---|---:|---:|
| Cold prefill @ 127K | 3.14× slower | 1.00× |
| Decode @ 127K | parity, 1.03× | 1.00× |

Measurements, method, and every killed idea: [`docs/campaign-scorecard.md`](docs/campaign-scorecard.md).
