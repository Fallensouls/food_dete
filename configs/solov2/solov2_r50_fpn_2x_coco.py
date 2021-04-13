_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# import SOLOv2Head, MaskFeatHead and SOLOv2
custom_imports = dict(
    imports=['models.dense_heads.solov2_head',
             'models.detectors.solov2', 'models.mask_heads.mask_feat_head'],
    allow_failed_imports=False)

model = dict(
    type='SOLOv2',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # C2, C3, C4, C5
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='SOLOv2Head',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=512,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
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
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    mask_feat_head=dict(
        type='MaskFeatHead',
        in_channels=256,
        out_channels=128,
        start_level=0,
        end_level=3,
        mask_feat_channels=256,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        update_thr=0.05,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100)
)

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
