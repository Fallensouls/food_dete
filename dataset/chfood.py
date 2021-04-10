import numpy as np
import os
import json
from pycocotools import mask as mask_util
import mmcv
from PIL import Image


def get_categories(data_path):
    classes = list(
        json.load(open(os.path.join(data_path, "classes_en.json"))).keys())
    categories = []
    for idx, class_name in enumerate(classes):
        categories.append({'id': idx, 'name': class_name})
    return categories


def get_categories_tuple(data_path):
    classes = list(
        json.load(open(os.path.join(data_path, "classes_en.json"))).keys())
    return tuple(classes)


def convert_chfood_to_coco(root, dataset_name='train'):
    if dataset_name == 'train':
        img_dir = "train_images"
        mask_dir = "train_mask"
        out_file = 'chfood_train_coco_format.json'
    else:
        img_dir = "test_images"
        mask_dir = "test_mask"
        out_file = 'chfood_test_coco_format.json'

    img_dir = os.path.join(root, img_dir)
    mask_dir = os.path.join(root, mask_dir)
    imgs = list(os.listdir(img_dir))
    masks = list(os.listdir(os.path.join(root, mask_dir)))

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


# convert_chfood_to_coco("/home/hatsunemiku/dev/food_dete/data/ch_food")
# print(get_categories_tuple("/home/hatsunemiku/dev/food_dete/data/ch_food"))
