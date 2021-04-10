import numpy as np
import os
from pycocotools import mask as mask_util
import mmcv
from PIL import Image


def get_categories(data_path):
    with open(os.path.join(data_path, 'labels.txt'), 'r') as f:
        classes = f.readlines()
    f.close()
    classes = [x.strip() for x in classes][2:]
    categories = []
    for idx, class_name in enumerate(classes):
        categories.append({'id': idx, 'name': class_name})
    return categories


def get_categories_tuple(data_path):
    with open(os.path.join(data_path, 'labels.txt'), 'r') as f:
        classes = f.readlines()
    f.close()
    classes = [x.strip() for x in classes][2:]
    return tuple(classes)


def convert_food201_to_coco(root, dataset_name='test'):
    if dataset_name == 'train':
        img_dir = os.path.join(root, "segmented_train")
        mask_dir = os.path.join(root, "new_masks_train")
        with open(os.path.join(root, 'train_pixel_annotations.txt'), 'r') as f:
            lines = f.readlines()
        masks = [line.strip() for line in lines]
        imgs = [line.strip().split('.')[0]+'.jpg' for line in lines]
        out_file = 'food201_train_coco_format.json'
    else:
        img_dir = os.path.join(root, "segmented_test")
        mask_dir = os.path.join(root, "new_masks_test")
        with open(os.path.join(root, 'test_pixel_annotations.txt'), 'r') as f:
            lines = f.readlines()
        masks = [line.strip()[23:] for line in lines]
        imgs = [line.strip().split('.')[0][23:] +
                '.jpg' for line in lines]
        out_file = 'food201_test_coco_format.json'

    annotations = []
    images = []
    obj_count = 0
    for idx, img in enumerate(imgs):
        img_path = os.path.join(img_dir, img)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=img,
            height=height,
            width=width))

        mask_path = os.path.join(mask_dir, masks[idx])
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:] if obj_ids[0] == 0 else obj_ids

        # split the color-encoded mask into a set
        # of binary masks
        mask_list = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.where(mask_list[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            segm = mask_util.encode(
                np.asarray(mask_list[i], dtype=np.uint8, order="F"))
            segm['counts'] = segm['counts'].decode('ascii')
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=obj_ids[i]-1,
                bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                area=(xmax - xmin) * (ymax - ymin),
                segmentation=segm,
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=get_categories(root))
    mmcv.dump(coco_format_json, os.path.join(root, out_file))


# convert_food201_to_coco("/home/hatsunemiku/dev/food_dete/data/food201")
# print(get_categories_tuple("/home/hatsunemiku/dev/food_dete/data/food201"))
