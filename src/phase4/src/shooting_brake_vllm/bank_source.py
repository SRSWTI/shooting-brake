"""Shared host-side bank source: pageable np.memmap or registered mmap.

Used by both the Marlin (bank v1) and B12x (bank v2) prefill streamers.
See marlin_prefill.py for the design history; measured numbers in
benchmarks/results/slab_h2d/hostregister_probe.json (53.9 GiB/s DMA from
registered page cache vs 18.5 GiB/s pageable).
"""

from __future__ import annotations

import mmap
import time

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class BankSource:
    """uint8 tensor over a bank file's plane data, pageable or pinned.

    Keeps the mmap object alive for the tensor's lifetime. Registration
    (``register=True``) pins the file-backed page cache with
    cudaHostRegister so non_blocking H2D copies run as true async DMA at
    the ceiling. MUST be constructed in the worker process (post-fork):
    pinned MAP_PRIVATE pages + fork() is the classic get_user_pages
    hazard. Hard-fails on register error -- a failed cudaHostRegister
    poisons the next CUDA runtime check in this process (measured), so a
    silent pageable fallback would serve corrupted-context behaviour.
    """

    def __init__(self, path: str, data_offset: int, data_len: int,
                 register: bool) -> None:
        self._registered_ptr: int | None = None
        if not register:
            self._mm = np.memmap(path, dtype=np.uint8, mode="r")
            self.tensor = torch.from_numpy(
                np.asarray(self._mm[data_offset: data_offset + data_len])
            )
            return

        import os
        map_len = data_offset + data_len
        fd = os.open(path, os.O_RDONLY)
        try:
            # MAP_PRIVATE|PROT_WRITE over an O_RDONLY fd: COW reserves the
            # write option and never fires (we never write); sidesteps
            # cudaHostRegisterReadOnly and its device-attribute lottery.
            self._mmap = mmap.mmap(
                fd, map_len, flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)
        self._mmap.madvise(mmap.MADV_WILLNEED)
        base = torch.frombuffer(
            memoryview(self._mmap)[data_offset: data_offset + data_len],
            dtype=torch.uint8,
        )
        # data_offset and layer strides are ALIGNMENT(4096)-aligned by both
        # bank ABIs, so the registered range is page-aligned by construction.
        t0 = time.perf_counter()
        status = int(torch.cuda.cudart().cudaHostRegister(
            base.data_ptr(), data_len, 0
        ))
        if status != 0:
            raise RuntimeError(
                f"cudaHostRegister({data_len} B) failed with cudaError "
                f"{status} -- refusing pageable fallback: the failed call "
                "poisons the CUDA context."
            )
        logger.info(
            "Registered %.1f GiB bank page cache in %.1f s",
            data_len / 2**30, time.perf_counter() - t0,
        )
        self._registered_ptr = base.data_ptr()
        self.tensor = base
