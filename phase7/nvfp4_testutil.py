"""Shared NVFP4 fixtures for the CPU-tier gates.

Three gates need the same two things -- a packed plane to load, and a
trustworthy dequantization of those exact bytes to compare against -- so they
live here rather than being reinvented per test with slightly different
conventions, which is the failure mode this tier is most exposed to.

The reference is vLLM's own ``dequantize_to_dtype``. That choice is the point:
the arena claims to evaluate ``e2m1(nibble) * e4m3(blockscale) * gscale``, and
the only convincing evidence is agreeing with the implementation the CUDA path
already trusts. Deriving a reference independently would just re-guess the
conventions -- notably the global scale, which vLLM's reference *quantiser*
folds in the opposite direction from its dequantizer.

Bytes are generated directly rather than by quantizing random weights, which
covers nibble codes and scale exponents a real checkpoint might never emit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4" / "src"))

from shooting_brake_vllm.cpu_expert_host import PackedPlane  # noqa: E402

BLOCK = 16


def make_plane(
    rows: int, cols: int, gen: torch.Generator, gscale: float = 1.0
) -> PackedPlane:
    """A packed NVFP4 plane spanning every nibble code and a sane scale band.

    Nibbles are uniform over all 16 codes, so both signs, zero, the subnormal
    0.5 and the top magnitude 6 all appear. Scale bytes keep exponent in
    [4, 10]: below that e4m3 subnormals are vanishingly small and would swamp
    any comparison in relative terms, and exponent 15 with mantissa 7 is NaN.
    """
    if cols % BLOCK:
        raise ValueError(f"cols={cols} must be a multiple of {BLOCK}")
    q = torch.randint(0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8)
    exp = torch.randint(4, 11, (rows, cols // BLOCK), generator=gen)
    man = torch.randint(0, 8, (rows, cols // BLOCK), generator=gen)
    sf = ((exp << 3) | man).to(torch.uint8)
    return PackedPlane(q.contiguous(), sf.contiguous(), gscale)


def dequant(plane: PackedPlane, rows: int, cols: int) -> torch.Tensor:
    """fp32 ground truth for ``plane``, via vLLM, from the identical bytes.

    ``swizzle=False`` is the matching entry point -- the arena stores linear
    block scales, having undone the checkpoint's swizzle at load time. That
    path is a Triton kernel and so runs on device when one is available.
    """
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
        dequantize_to_dtype,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dequantize_to_dtype(
        plane.q.to(dev),
        plane.sf.to(dev),
        torch.tensor(plane.gscale, dtype=torch.float32, device=dev),
        torch.float32,
        block_size=BLOCK,
        swizzle=False,
    ).reshape(rows, cols).cpu()


def make_expert(
    hidden: int, inter: int, gen: torch.Generator
) -> tuple[tuple[PackedPlane, PackedPlane, PackedPlane],
           tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One expert as (packed planes, dequantized references).

    Returns both because every caller needs both: the packed form to load and
    the dequantized form to check against.
    """
    g = make_plane(inter, hidden, gen)
    u = make_plane(inter, hidden, gen)
    d = make_plane(hidden, inter, gen)
    return (g, u, d), (dequant(g, inter, hidden),
                       dequant(u, inter, hidden),
                       dequant(d, hidden, inter))


def ffn(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
        down: torch.Tensor) -> torch.Tensor:
    """SwiGLU reference in fp32: ``(silu(x@gate^T) * (x@up^T)) @ down^T``."""
    xf = x.float()
    return (torch.nn.functional.silu(xf @ gate.T) * (xf @ up.T)) @ down.T
