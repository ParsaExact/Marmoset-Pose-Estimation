"""Fine-tune MarmoPose's RTMPose model on in-domain data.

Differences from the shipped `models/pose_model/config.py`, and why:

* `dataset_type = 'CocoDataset'` with an explicit `metainfo`. The shipped config
  names `MarmosetDataset`, which is not registered in mmpose 1.3.1 -- it lived in
  the authors' own tree -- so training with that config fails outright. mmpose's
  generic topdown CocoDataset accepts any keypoint set as long as it is handed the
  metainfo, and `models/pose_model_deployed/data_meta.py` already defines all 16
  keypoints with their left/right swap pairs (needed for flip augmentation).

* The `Albumentation` transform is dropped. mmpose 1.3.1's wrapper targets
  albumentations 1.x, and installing albumentations 2.x drags in numpy 2 and
  opencv 5, which break xtcocotools and scikit-video in this environment. Its
  blur/dropout augs matter far less than the domain adaptation being done here,
  and `YOLOXHSVRandomAug` still covers colour jitter.

* Low LR, few epochs, `load_from` the pretrained checkpoint. This is adaptation of
  an already-trained model to a new cage/lighting, not training from scratch, so
  the schedule is short and gentle to avoid washing out what the model knows.

Override the data paths on the command line, e.g.
    python tools/finetune.py configs/finetune_pose.py \
        --cfg-options data_root=datasets/snuh_pseudo/ max_epochs=40
"""
num_keypoints = 16
input_size = (512, 512)

# Where the metainfo (keypoint names, flip pairs, colours) comes from.
metainfo = dict(from_file='models/pose_model_deployed/data_meta.py')

codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(12, 12),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# Start from the shipped weights -- this is fine-tuning, not fresh training.
load_from = 'models/pose_model/best.pth'

dataset_type = 'CocoDataset'
data_root = 'datasets/demo_pseudo/'
data_mode = 'topdown'
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.1),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

test_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.1),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# `data_prefix=dict(img='')` because the exported COCO `file_name` already
# carries the `images/` prefix.
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val.json')
test_evaluator = val_evaluator

max_epochs = 40
base_lr = 2.0e-4          # ~20x below the from-scratch 0.004
interval = 5

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=interval)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-3, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2,
         end=max_epochs, T_max=max_epochs // 2, by_epoch=True,
         convert_to_iter_based=True),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', save_best='coco/AP', rule='greater',
                    interval=interval, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))

custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=1e-4,
         update_buffers=True, priority=49),
]

auto_scale_lr = dict(enable=False, base_batch_size=16)
randomness = dict(seed=21)

default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    # 'fork' does not exist on Windows, and nccl is Linux-only.
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends,
                  name='visualizer')
log_processor = dict(by_epoch=True, num_digits=6, window_size=20)
log_level = 'INFO'
