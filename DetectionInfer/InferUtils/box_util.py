
## nms.py
import numpy as np


def get_iou(boxes1, boxes2):
    """
    :param boxes1: [n, 4], (x1, y1, x2, y2)
    :param boxes2: [m, 4]
    :return: iou, [n, m]
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    # 将boxes的每个box复制M遍，对应于与boxes2的每个box求iou，相当于将boxes复制M遍
    boxes1 = np.repeat(np.expand_dims(boxes1, axis=1), m, axis=1)  # [n, m, 4]
    # boxes2同理
    boxes2 = np.repeat(np.expand_dims(boxes2, axis=0), n, axis=0)  # [n, m, 4]
    # 求取boxes1和boxes2的每个box相互之间的交点
    # 左上取最大, 右下取最小
    x1y1 = np.maximum(boxes1[:, :, :2], boxes2[:, :, :2])
    x2y2 = np.minimum(boxes1[:, :, 2:], boxes2[:, :, 2:])
    # 算宽和高，由于可能某两个box没有重叠，此时通过用x2y2-x1y1可能会出现负数，这时需要截取
    wh = np.clip(x2y2 - x1y1, a_min=0, a_max=None)  # shape：[n, m, 2]
    # 计算交集的面积
    area_inter = wh[:, :, 0] * wh[:, :, 1]  # shape：[n, m]
    # 计算每个box的面积，再求并集的面积
    area_boxes1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])  # shape：[n, m]
    area_boxes2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])  # shape：[n, m]
    area_union = area_boxes1 + area_boxes2 - area_inter
    return area_inter / area_union

def box_nms(boxes, conf, iou_thresh=0.45, max_det=300):
    '''
    boxes: [n, 4], xyxy
    conf: [n,]
    '''
    # 进行nms
    index = np.argsort(conf)[::-1]  # 排序，分数由大到小的序号
    # 遍历index
    keep = []  # 要保存的box的序号
    while index.shape[0] > 0:
        idx = index[0]
        keep.append(idx)  # 当前最高分的一定保留
        if index.shape[0] == 1:  # 如果只剩下这一个，就不用在继续了，否则要计算iou进行筛选
            break
        # 当前序号的box和后续序号的box的iou,返回结果shape为[k]
        iou = get_iou(np.expand_dims(boxes[index[0], :], axis=0), boxes[index[1:], :]).squeeze(0)
        new_index = iou < iou_thresh  # iou<thresh的保留，其余舍弃
        index = index[1:][new_index]  # 更新index.index[0]代表当前序号，所以要从index[1:]选取iou满足要求的来更新
        if len(keep) >= max_det:
            break
    return keep

def nms(results: dict, conf_thresh=0.25, iou_thresh=0.45, max_det=300): 
    '''
    boxes: [n, 4], xywh
    conf: [n, 1]
    class_ids: [n, 1]
    '''
    confs, class_ids, boxes, points = results["confs"], results["class_ids"], \
            results["boxes"], results.get("points")
    conf_mask = confs.squeeze(-1) > conf_thresh
    confs = confs[conf_mask].squeeze(-1)
    class_ids = class_ids[conf_mask]
    boxes = boxes[conf_mask]
    points = points[conf_mask] if points is not None else None
    # xywh2xyxy
    boxes[..., :2] -= boxes[..., 2:] / 2
    boxes[..., 2:] += boxes[..., :2]
    keep = box_nms(boxes, confs, iou_thresh, max_det)
    results.update(confs=confs[keep], class_ids=class_ids[keep], boxes=boxes[keep], 
                   points=points[keep] if points is not None else None)
    return results