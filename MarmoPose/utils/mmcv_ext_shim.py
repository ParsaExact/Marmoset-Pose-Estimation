"""Stand in for mmcv's compiled ``mmcv._ext`` extension.

Why this exists
---------------
RTX 5090 (Blackwell, sm_120) needs CUDA 12.8, which in turn needs torch >= 2.7.
No prebuilt ``mmcv`` wheel is published for that combination, and building one
from source needs an nvcc/MSVC pair that doesn't exist on Windows yet. So we
install ``mmcv-lite``, which ships every ``mmcv.ops`` Python module but not the
compiled ``mmcv._ext`` they load at import time.

That missing extension is the only thing standing between mmdet/mmpose and a
working GPU. MarmoPose runs RTMDet for detection and RTMPose (SimCC/RTMCC) for
keypoints; between them the single compiled op ever *executed* is non-maximum
suppression. Every other symbol -- deformable conv, RoIAlign, CARAFE, corner
pooling -- belongs to architectures this project never instantiates, and is only
touched because ``mmcv/ops/__init__.py`` imports the whole package eagerly.

So this module registers a fake ``mmcv._ext`` that
  * implements ``nms``, ``softnms`` and ``nms_match`` for real, and
  * hands back a raising placeholder for anything else.

``mmcv.ops.nms`` then routes through torchvision's NMS kernel, which ships
Blackwell binaries and matches mmcv's own implementation semantically (mmcv's
NMS is itself derived from torchvision's). Detection stays entirely on the GPU.

Import this module *before* mmdet/mmpose. It is a no-op when a genuine compiled
``mmcv._ext`` is present, so an environment with a full mmcv build is unaffected.
"""
import importlib.machinery
import importlib.util
import sys
import types

import torch
from torchvision.ops import nms as _tv_nms

__all__ = ['install']

# Ops we cannot emulate are still importable; they only fail if actually called.
_MESSAGE = (
    "mmcv operator '{name}' needs the compiled mmcv._ext extension, which is not "
    'installed. MarmoPose does not use this operator -- reaching it means the model '
    'config asks for an architecture beyond RTMDet/RTMPose. Build mmcv from source '
    'if you need it.')


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float,
         offset: int) -> torch.Tensor:
    """Indices kept by NMS, ordered by descending score.

    ``offset`` is mmcv's convention for box width: 0 means ``x2 - x1``, 1 means
    ``x2 - x1 + 1``. torchvision only implements the former, so widen the boxes
    by one pixel to reproduce the latter's IoU exactly.
    """
    boxes = boxes.float()
    if offset == 1:
        boxes = boxes + boxes.new_tensor([0.0, 0.0, 1.0, 1.0])
    return _tv_nms(boxes, scores.float(), float(iou_threshold))


def _softnms(boxes: torch.Tensor, scores: torch.Tensor, dets: torch.Tensor,
             iou_threshold: float, sigma: float, min_score: float, method: int,
             offset: int) -> torch.Tensor:
    """Soft-NMS, writing surviving boxes into ``dets`` as mmcv's op does.

    ``method`` follows mmcv: 0 hard, 1 linear, 2 gaussian. Runs on CPU, which is
    where mmcv's own soft-NMS runs too -- ``SoftNMSop.forward`` moves every
    tensor to CPU before calling the extension.
    """
    boxes = boxes.cpu().float()
    scores = scores.cpu().float().clone()
    areas_off = 1.0 if offset == 1 else 0.0

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1 + areas_off) * (y2 - y1 + areas_off)
    order = torch.arange(boxes.size(0))

    keep = []
    while order.numel() > 0:
        top = int(torch.argmax(scores[order]))
        idx = order[top]
        keep.append(int(idx))
        order = torch.cat((order[:top], order[top + 1:]))
        if order.numel() == 0:
            break

        xx1 = torch.maximum(x1[idx], x1[order])
        yy1 = torch.maximum(y1[idx], y1[order])
        xx2 = torch.minimum(x2[idx], x2[order])
        yy2 = torch.minimum(y2[idx], y2[order])
        inter = ((xx2 - xx1 + areas_off).clamp(min=0) *
                 (yy2 - yy1 + areas_off).clamp(min=0))
        iou = inter / (areas[idx] + areas[order] - inter)

        if method == 0:
            weight = (iou <= iou_threshold).float()
        elif method == 1:
            weight = torch.where(iou > iou_threshold, 1 - iou,
                                 torch.ones_like(iou))
        else:
            weight = torch.exp(-(iou * iou) / sigma)
        scores[order] *= weight

        order = order[scores[order] > min_score]

    inds = torch.tensor(keep, dtype=torch.int64)
    out = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    # mmcv's extension fills `dets` in place and returns the kept indices.
    dets.resize_(out.shape).copy_(out)
    return inds


def _nms_match(dets: torch.Tensor, iou_threshold: float):
    """Group boxes into clusters, each represented by its highest-scoring box."""
    dets = dets.cpu().float()
    boxes, scores = dets[:, :4], dets[:, 4]
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1) * (y2 - y1)
    order = torch.argsort(scores, descending=True)

    groups = []
    while order.numel() > 0:
        idx = order[0]
        rest = order[1:]
        if rest.numel() == 0:
            groups.append(torch.tensor([int(idx)], dtype=torch.int64))
            break

        xx1 = torch.maximum(x1[idx], x1[rest])
        yy1 = torch.maximum(y1[idx], y1[rest])
        xx2 = torch.minimum(x2[idx], x2[rest])
        yy2 = torch.minimum(y2[idx], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[idx] + areas[rest] - inter)

        matched = iou > iou_threshold
        groups.append(torch.cat((idx.reshape(1), rest[matched])))
        order = rest[~matched]

    return groups


_IMPLEMENTED = {
    'nms': _nms,
    'softnms': _softnms,
    'nms_match': _nms_match,
}


def _make_placeholder(name: str):
    def _missing(*args, **kwargs):
        raise NotImplementedError(_MESSAGE.format(name=name))

    _missing.__name__ = name
    return _missing


def _module_getattr(name: str):
    """Resolve any symbol mmcv's ext_loader asks for.

    ``ext_loader.load_ext`` asserts ``hasattr(ext, fun)`` for every op a module
    declares, so this has to answer for names we've never heard of. PEP 562
    module ``__getattr__`` covers ``hasattr`` too.
    """
    if name.startswith('__'):
        raise AttributeError(name)
    return _IMPLEMENTED.get(name) or _make_placeholder(name)


def install() -> bool:
    """Register the fake ``mmcv._ext``. Returns True if the shim was installed.

    Does nothing when a real compiled extension exists, or when the shim is
    already registered.
    """
    if 'mmcv._ext' in sys.modules:
        return getattr(sys.modules['mmcv._ext'], '__marmopose_shim__', False)

    try:
        if importlib.util.find_spec('mmcv._ext') is not None:
            return False  # a genuine compiled build is installed; leave it alone
    except (ImportError, ValueError):
        pass

    ext = types.ModuleType('mmcv._ext')
    ext.__marmopose_shim__ = True
    ext.__getattr__ = _module_getattr
    # mmengine's mmcv_full_available() probes this via pkgutil.find_loader, which
    # raises rather than returning False when __spec__ is None. Its only inference
    # -time caller is revert_sync_batchnorm, which just isinstance-checks against
    # mmcv.ops.SyncBatchNorm; the rest are training optimizer constructors.
    ext.__spec__ = importlib.machinery.ModuleSpec('mmcv._ext', loader=None)
    for op_name, fn in _IMPLEMENTED.items():
        setattr(ext, op_name, fn)
    sys.modules['mmcv._ext'] = ext
    return True
