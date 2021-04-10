# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=23),
        mask_head=dict(num_classes=23)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Rice', 'White radish', 'Spinach', 'Fried cabbage', 'Scrambled eggs', 'Tofu', 'Carob', 'Bean sprouts', 'Snow peas', 'Red pepper', 'Carrot',
           'Cucumber', 'Chicken', 'Green pepper', 'Roast pork', 'Onion', 'Chicken feet', 'Roast Duck', 'Sausage', 'Tripe', 'Beef tendon', 'Pig ear', 'Pig feet')

data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/ch_food/train_images/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/ch_food/chfood_train_coco_format.json'),
    val=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/ch_food/test_images/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/ch_food/chfood_test_coco_format.json'),
    test=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/ch_food/test_images/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/ch_food/chfood_test_coco_format.json'))

checkpoint_config = dict(interval=4)
workflow = [('train', 1), ('val', 1)]
evaluation = dict(metric=['bbox', 'segm'], proposal_nums=(1, 10, 100))
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
