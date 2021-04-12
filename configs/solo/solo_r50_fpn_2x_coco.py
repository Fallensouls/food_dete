_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['models.dense_heads.solo_head',
             'models.detectors.solo'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='SOLO',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='SOLOHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=7,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        loss_mask=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    ))
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
