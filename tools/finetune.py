"""Fine-tune either MarmoPose model (RTMPose keypoints or RTMDet detection).

One entry point for both, rather than mmpose's/mmdet's own `tools/train.py`,
because a few things have to be fixed up before the Runner starts:

  * `mmcv._ext` has to be shimmed before mmdet/mmpose import (RTX5090_SETUP.md)
  * mmengine's checkpoint loader predates torch 2.6's `weights_only` default
  * SyncBN has no process group to sync over on a single GPU
  * config paths are repo-relative, but the Runner resolves against the cwd

Usage:
    python tools/finetune.py configs/finetune_pose.py
    python tools/finetune.py configs/finetune_det.py --cfg-options data_root=datasets/snuh_pseudo/
    python tools/finetune.py configs/finetune_pose.py --smoke-test
"""
import argparse
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Must precede any mmdet/mmpose import.
from marmopose.utils.mmcv_ext_shim import install as install_shim
install_shim()
from marmopose.utils.torch_compat import legacy_torch_load  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='e.g. configs/finetune_pose.py or configs/finetune_det.py')
    ap.add_argument('--work-dir', default=None)
    ap.add_argument('--resume', action='store_true', default=False)
    ap.add_argument('--cfg-options', nargs='+', default=None,
                    help='override config entries as key=value')
    ap.add_argument('--smoke-test', action='store_true', default=False,
                    help='one short epoch to prove the loop works, then stop')
    return ap.parse_args()


def _coerce(raw):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def apply_overrides(cfg, overrides):
    """Apply `key=value` overrides, propagating data_root into dependent entries."""
    for item in overrides or []:
        key, _, raw = item.partition('=')
        value = _coerce(raw)
        cfg[key] = value
        if key == 'data_root':
            for loader in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
                if loader in cfg and 'dataset' in cfg[loader]:
                    cfg[loader]['dataset']['data_root'] = value
            for evaluator in ('val_evaluator', 'test_evaluator'):
                if evaluator in cfg and 'ann_file' in cfg[evaluator]:
                    cfg[evaluator]['ann_file'] = value + 'val.json'


def replace_syncbn(node):
    """Swap SyncBN for BN throughout the model config.

    SyncBN needs a distributed process group; on one GPU the Runner would fail.
    RTMDet uses it in the backbone, the neck and the head, so recurse rather than
    patching known keys.
    """
    if isinstance(node, dict):
        if node.get('type') == 'SyncBN':
            node['type'] = 'BN'
        for value in node.values():
            replace_syncbn(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            replace_syncbn(value)


def absolutize(cfg):
    """Make repo-relative paths absolute so the cwd doesn't matter."""
    for loader in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        dataset = cfg.get(loader, {}).get('dataset')
        if dataset and not Path(str(dataset.get('data_root', ''))).is_absolute():
            dataset['data_root'] = str(REPO / dataset['data_root']) + '/'
    for evaluator in ('val_evaluator', 'test_evaluator'):
        entry = cfg.get(evaluator)
        if entry and 'ann_file' in entry and not Path(entry['ann_file']).is_absolute():
            entry['ann_file'] = str(REPO / entry['ann_file'])
    if cfg.get('load_from') and not Path(cfg.load_from).is_absolute():
        cfg.load_from = str(REPO / cfg.load_from)

    metainfo = cfg.get('metainfo')
    if isinstance(metainfo, dict) and 'from_file' in metainfo:
        if not Path(metainfo['from_file']).is_absolute():
            metainfo['from_file'] = str(REPO / metainfo['from_file'])
        for loader in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            if loader in cfg and 'dataset' in cfg[loader]:
                cfg[loader]['dataset']['metainfo'] = metainfo


def shrink_for_smoke_test(cfg):
    cfg.train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
    for loader in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        if loader in cfg:
            cfg[loader]['batch_size'] = 2
            cfg[loader]['num_workers'] = 0
            cfg[loader]['persistent_workers'] = False  # invalid with 0 workers
    cfg.default_hooks['logger']['interval'] = 1
    cfg.custom_hooks = []
    cfg.work_dir = str(REPO / 'work_dirs' / '_smoke')


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    from mmengine.config import Config
    from mmengine.runner import Runner

    config_path = Path(args.config)
    cfg = Config.fromfile(str(config_path if config_path.is_absolute() else REPO / config_path))
    apply_overrides(cfg, args.cfg_options)

    cfg.work_dir = args.work_dir or str(REPO / 'work_dirs' / config_path.stem)
    cfg.resume = args.resume

    absolutize(cfg)
    replace_syncbn(cfg.model)

    if args.smoke_test:
        shrink_for_smoke_test(cfg)

    with legacy_torch_load():
        runner = Runner.from_cfg(cfg)
        runner.train()


if __name__ == '__main__':
    main()
