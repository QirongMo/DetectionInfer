
## Yolov8OnnxInfer.py
from .Yolov5OnnxInfer import Yolov5OnnxInfer
import numpy as np


class Yolov10OnnxInfer(Yolov5OnnxInfer):
    def __init__(self, model_config={}):
        super().__init__(model_config)

    def decode_result(self, pred_results):
        boxes = pred_results[..., :4]
        boxes = np.concatenate([(boxes[..., 2:4] + boxes[..., :2])/2, boxes[..., 2:4] - boxes[..., :2]], axis=1)
        class_conf = pred_results[..., 4:5]
        class_ids = pred_results[..., 5:6]
        # keep = [i for i, conf in enumerate(class_conf) if conf > 0.3]
        # boxes = boxes[keep]
        # class_conf = class_conf[keep]
        # class_ids = class_ids[keep]
        results = {
            "confs": class_conf,
            "class_ids": class_ids,
            "boxes": boxes,
            "points": None,
        }
        return results
