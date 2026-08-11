"""Make mmengine checkpoints loadable under PyTorch >= 2.6.

PyTorch 2.6 flipped ``torch.load``'s ``weights_only`` default from False to True.
mmengine 0.10.7 predates that and calls ``torch.load`` with no such argument, so
loading any MarmoPose ``.pth`` now fails with ``UnpicklingError``.

Allowlisting types via ``torch.serialization.add_safe_globals`` is the tidy fix
where it works, but it doesn't here: these checkpoints pickle a ``meta`` block of
training history whose reconstruction needs the builtin ``getattr``. Allowlisting
``getattr`` would hand back arbitrary attribute access and make ``weights_only``
protect nothing -- worse than turning it off, because it would look safe.

So we do the plain thing instead and restore the pre-2.6 behaviour, narrowly: only
around the calls that load MarmoPose's own bundled checkpoints, never process-wide.
Everything else in the process keeps the torch 2.6 default.

Needed because an RTX 5090 forces torch >= 2.7 (see ``mmcv_ext_shim``); the pinned
torch 2.4 stack in the older ``marmopose`` env never hit this.
"""
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


@contextmanager
def legacy_torch_load():
    """Run a block with ``torch.load`` defaulting to ``weights_only=False``.

    For loading trusted local checkpoints through third-party code that predates
    torch 2.6 and cannot pass the argument itself. An explicit ``weights_only``
    from the caller still wins.

    This swaps a module-level attribute, so it is not thread-safe; keep the block
    small and confined to model construction.
    """
    original = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original(*args, **kwargs)

    torch.load = _load
    try:
        yield
    finally:
        torch.load = original
