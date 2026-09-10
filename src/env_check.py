"""
env_check.py -- hard GPU / CUDA precondition check.

The project brief is explicit: training must run on the RTX 4060 via CUDA, and
if CUDA is missing we want a LOUD failure, not a silent 40x-slower CPU run.
Every entry point that is about to move tensors onto a device
(``train.py``, ``main.py``) calls :func:`require_cuda` first.

Run standalone to print an environment report:

    python src/env_check.py
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers only; no runtime import cost
    import torch


class CudaUnavailableError(RuntimeError):
    """Raised when a CUDA device is required but not usable."""


def require_cuda(verbose: bool = True) -> "torch.device":
    """
    Assert that a usable CUDA GPU is present and return ``torch.device('cuda')``.

    Raises
    ------
    CudaUnavailableError
        If torch has no CUDA support compiled in, or no GPU is visible, or a
        trivial GPU tensor allocation fails. The message spells out the likely
        cause so it is actionable.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise CudaUnavailableError(f"PyTorch import failed: {exc!r}") from exc

    if not torch.cuda.is_available():
        raise CudaUnavailableError(
            "torch.cuda.is_available() is False.\n"
            "  * Is this the CUDA build of torch?  "
            f"(installed: {torch.__version__} -- a '+cpu' suffix means CPU-only)\n"
            "  * Is an NVIDIA driver present and recent enough for this CUDA "
            "runtime?\n"
            "Refusing to continue on CPU -- see requirements.txt for the CUDA "
            "install command."
        )

    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise CudaUnavailableError(
            "CUDA reports as available but device_count() == 0."
        )

    # Actually touch the GPU: a driver/runtime mismatch often only shows up on
    # the first allocation, not on is_available().
    try:
        probe = torch.zeros(8, device="cuda")
        _ = (probe + 1).sum().item()
        del probe
        torch.cuda.synchronize()
    except Exception as exc:
        raise CudaUnavailableError(
            f"A GPU is visible but a test allocation failed: {exc!r}"
        ) from exc

    device = torch.device("cuda")
    if verbose:
        name = torch.cuda.get_device_name(0)
        cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("=" * 60)
        print("CUDA precondition check: PASS")
        print(f"  torch            : {torch.__version__}")
        print(f"  torch CUDA build : {torch.version.cuda}")
        print(f"  visible GPUs     : {device_count}")
        print(f"  using GPU 0      : {name}")
        print(f"  compute capab.   : sm_{cap.replace('.', '')}")
        print(f"  total VRAM       : {total_gb:.1f} GiB")
        print("=" * 60)
    return device


def _main() -> None:
    try:
        require_cuda(verbose=True)
    except CudaUnavailableError as exc:
        print(f"CUDA precondition check: FAIL\n\n{exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    _main()
