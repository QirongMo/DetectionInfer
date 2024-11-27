
## Yolov8OnnxInfer.py
from .Yolov5OnnxInfer import Yolov5OnnxInfer
import numpy as np


class Yolov8OnnxInfer(Yolov5OnnxInfer):
    def __init__(self, model_config={}):
        super().__init__(model_config)

    def decode_result(self, pred_results):
        pred_results = pred_results.transpose((1, 0))
        dims = pred_results.shape[1]
        boxes = pred_results[..., :4]
        anchor_conf = pred_results[..., 4:4+len(self.class_names)]
        anchor_max_conf = np.max(anchor_conf, axis=1, keepdims=True)
        class_ids = np.argmax(anchor_conf, axis=1, keepdims=True)
        class_conf = anchor_max_conf
        points = pred_results[..., 4+len(self.class_names):] if dims > 4+len(self.class_names) else None
        results = {
            "confs": class_conf,
            "class_ids": class_ids,
            "boxes": boxes,
            "points": points,
        }
        return results
