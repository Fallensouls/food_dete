import numpy as np
import skimage
import os
import json
from pycocotools import mask as mask_util
import mmcv


def get_categories(data_path):
    classes = list(
        json.load(open(os.path.join(data_path, "classes.json"))).keys())
    categories = []
    for idx, class_name in enumerate(classes):
        categories.append({'id': idx, 'name': class_name})
    return categories


def get_categories_tuple(data_path):
    classes = list(
        json.load(open(os.path.join(data_path, "classes.json"))).keys())
    return tuple(classes)


def gen_mask(polygons, width, height):
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [instance count, height, width] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    mask = np.zeros([len(polygons), height, width], dtype=np.uint8)

    for i, p in enumerate(polygons):
        # Get indexes of pixels inside the polygon and set them to 1
        points_y = p['all_points_y']
        points_x = p['all_points_x']
        rr, cc = skimage.draw.polygon(points_x, points_y)
        cc = width - cc
        mask[i, rr, cc] = 1

    # Return mask, and array of class IDs of each instance.
    return mask.astype(np.bool)


def get_label(annotations, classes, width, height):
    # Add images

    # Get the x, y coordinaets of points of the polygons that make up
    # the outline of each object instance. There are stores in the
    # shape_attributes (see json format above)
    polygons = [r['shape_attributes']
                for r in annotations['regions'].values()]

    categories = [r['region_attributes']['category']
                  for r in annotations['regions'].values()]

    # get the class_ids
    class_ids = []
    for c in categories:
        class_ids.append(int(classes[c])-1)

    masks = gen_mask(polygons, width, height)

    return class_ids, masks


def convert_unimib_to_coco(root, dataset_name='test'):
    if dataset_name == 'train':
        label_file = "train_region_data.json"
        out_file = 'unimib_train_coco_format.json'
    else:
        label_file = "test_region_data.json"
        out_file = 'unimib_test_coco_format.json'
    data_dir = os.path.join(root, dataset_name)
    classes = json.load(open(os.path.join(root, "classes.json")))
    raw_annotations = json.load(
        open(os.path.join(root, label_file)))
    imgs = list((os.listdir(data_dir)))

    annotations = []
    images = []
    obj_count = 0
    for idx, img in enumerate(imgs):
        img_path = os.path.join(data_dir, img)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=img,
            height=height,
            width=width))

        annotation = raw_annotations[img[:-4]]

        class_ids, masks = get_label(annotation, classes, width, height)

        num_objs = len(masks)

        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            segm = mask_util.encode(
                np.asarray(masks[i], dtype=np.uint8, order="F"))
            segm['counts'] = segm['counts'].decode('ascii')
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=class_ids[i],
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


# convert_unimib_to_coco("/home/hatsunemiku/dev/food_dete/data/UNIMIB2016")
