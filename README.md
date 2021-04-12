# FoodDete

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- mmcv-full
- mmdet

## Usage

### Train

```
python train.py configs/unimib/mask_rcnn_r50_fpn_2x_unimib.py
```

### Test

```
python test.py configs/unimib/mask_rcnn_r50_fpn_2x_unimib.py work_dirs/mask_rcnn_r50_fpn_2x_unimib/latest.pth --eval bbox segm
```

### Browse Dataset

```
python tools/browse_dataset.py configs/unimib/mask_rcnn_r50_fpn_2x_unimib.py
```

### Visualize

```
python tools/visualize.py configs/unimib/mask_rcnn_r50_fpn_2x_unimib.py work_dirs/mask_rcnn_r50_fpn_2x_unimib/latest.pth data/UNIMIB2016/test/20151211_131911.jpg
```

## Results (AP)

### UNIMIB-BBOX

|  Metric   | bbox_mAP | bbox_mAP_50 | bbox_mAP_75 | bbox_mAP_s | bbox_mAP_m | bbox_mAP_l |
| :-------: | :------: | :---------: | :---------: | :--------: | :--------: | :--------: |
| Mask RCNN |   82.3   |    92.0     |    89.2     |     -      |     -      |    82.3    |
|   Solo    |   83.1   |    91.0     |    87.5     |     -      |     -      |    83.1    |

### UNIMIB-SEGM

|  Metric   | segm_mAP | segm_mAP_50 | segm_mAP_75 | segm_mAP_s | segm_mAP_m | segm_mAP_l |
| :-------: | :------: | :---------: | :---------: | :--------: | :--------: | :--------: |
| Mask RCNN |   84.7   |    92.2     |    90.6     |     -      |     -      |    84.7    |
|   Solo    |   83.3   |    91.3     |    88.6     |     -      |     -      |    83.3    |

### chfood-BBOX

|  Metric   | bbox_mAP | bbox_mAP_50 | bbox_mAP_75 | bbox_mAP_s | bbox_mAP_m | bbox_mAP_l |
| :-------: | :------: | :---------: | :---------: | :--------: | :--------: | :--------: |
| Mask RCNN |  0.918   |    0.978    |    0.962    |  -1.0000   |  -1.0000   |   0.918    |
|  Solov2   |    -     |      -      |      -      |     -      |     -      |     -      |

### chfood-SEGM

|  Metric   | segm_mAP | segm_mAP_50 | segm_mAP_75 | segm_mAP_s | segm_mAP_m | segm_mAP_l |
| :-------: | :------: | :---------: | :---------: | :--------: | :--------: | :--------: |
| Mask RCNN |  0.884   |    0.981    |    0.981    |  -1.0000   |  -1.0000   |   0.884    |
|  Solov2   |    -     |      -      |      -      |     -      |     -      |     -      |

\* Results of this code were evaluated on 1 run.
