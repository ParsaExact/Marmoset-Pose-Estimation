"""Fine-tune MarmoPose's RTMDet detector on in-domain data.

Detection is ~38% of the 2D coverage shortfall on the SNUH cage: the shipped
detector boxes only 1.69 of 2 animals per frame there, versus 1.99 on MarmoPose's
own demo footage (see FINETUNE_2D.md). Every keypoint of an unboxed animal is lost
before the pose model even runs, so this has to be fixed alongside the pose model.

Inherits the shipped config and overrides only what has to change.

Two deliberate choices:

* **Classes are kept at 2, and every pseudo-label is written as class 0.** The
  shipped classes are dye colours (`white_head_marmoset`, `blue_head_marmoset`),
  and `heuristic_assign_bboxes` keeps at most one box per class -- which is exactly
  why one of the two animals gets silently dropped on this cage, where the animals
  are white and *pink*. The fix is to stop relying on the class for identity
  (`dye_identity: false` at inference) rather than to renumber the head: keeping
  `num_classes=2` lets the pretrained classification layer load cleanly instead of
  being discarded for a shape mismatch. Class 1 simply stops being predicted.

* **Mosaic and MixUp are dropped.** They are strong regularisers for training from
  scratch on a large set; on a few hundred in-domain frames they mostly manufacture
  implausible composites and slow convergence. The remaining resize/crop/flip
  augmentation is enough for domain adaptation.

Usage:
    python tools/finetune_det.py configs/finetune_det.py \
        --cfg-options data_root=datasets/snuh_pseudo/
"""
_base_ = ['../models/detection_model/config.py']

data_root = 'datasets/snuh_pseudo/'
img_scale = (640, 640)

# Start from the shipped detector.
load_from = 'models/detection_model/best.pth'

# Pseudo-label boxes come from the keypoint extent, so they are tight and honest,
# but there is no mask/crowd information -- keep the pipeline to geometry + colour.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# `data_prefix=dict(img='')` because the exported COCO file_name carries `images/`.
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=('white_head_marmoset', 'blue_head_marmoset')),
        ann_file='train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=('white_head_marmoset', 'blue_head_marmoset')),
        ann_file='val.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val.json',
                     metric='bbox', format_only=False)
test_evaluator = val_evaluator

max_epochs = 30
base_lr = 1.0e-4          # ~40x below the from-scratch 0.004
interval = 5

train_cfg = dict(max_epochs=max_epochs, val_interval=interval, dynamic_intervals=None)

optim_wrapper = dict(optimizer=dict(lr=base_lr))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-3, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2,
         end=max_epochs, T_max=max_epochs // 2, by_epoch=True,
         convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=interval, max_keep_ckpts=2,
                    save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=20))

# The shipped config swaps in a lighter pipeline for the last epochs via this hook;
# with mosaic gone there is nothing to switch to.
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))
