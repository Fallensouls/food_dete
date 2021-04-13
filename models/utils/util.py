import numpy as np


def segm2result(result, num_classes):
    if result is None:
        bbox_result = [np.zeros((0, 5), dtype=np.float32) for _ in
                       range(num_classes)]
        # BG is not included in num_classes
        segm_result = [[] for _ in range(num_classes)]
    else:
        bbox_result = [np.zeros((0, 5), dtype=np.float32) for _ in
                       range(num_classes)]
        segm_result = [[] for _ in range(num_classes)]
        seg_pred = result[0].detach().cpu().numpy()
        cate_label = result[1].detach().cpu().numpy()
        cate_score = result[2].detach().cpu().numpy()
        num_ins = seg_pred.shape[0]
        # extract bboxes from segmentation result
        bboxes = np.zeros((num_ins, 5), dtype=np.float32)
        bboxes[:, -1] = cate_score
        bboxes[:, :-1] = extract_bboxes(seg_pred)

        bbox_result = [bboxes[cate_label == i, :] for i in
                       range(num_classes)]

        for idx in range(num_ins):
            segm_result[cate_label[idx]].append(seg_pred[idx])
    return bbox_result, segm_result


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (x1, y1, x2, y2)].
    """
    num_ins = mask.shape[0]
    boxes = np.zeros([num_ins, 4], dtype=np.float32)
    for i in range(num_ins):
        m = mask[i, :, :]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes
