

# import pycuda.autoinit
from .Yolov5TrtInfer import Yolov5TrtInfer
import numpy as np


class Yolov8TrtInfer(Yolov5TrtInfer):
    def __init__(self, model_config={}):
        """
        :param model_config: model、label_list、thresh、nms_thresh
        """
        super().__init__(model_config)
       
    def decode_result(self, pred_results):
        boxes = pred_results[..., :4]
        anchor_conf = pred_results[..., 4:4+len(self.class_names)]
        anchor_max_conf = np.max(anchor_conf, axis=1, keepdims=True)
        class_ids = np.argmax(anchor_conf, axis=1, keepdims=True)
        class_conf = anchor_max_conf
        dims = pred_results.shape[1]
        points = pred_results[..., 4+len(self.class_names):] if dims > 4+len(self.class_names) else None
        results = {
            "confs": class_conf,
            "class_ids": class_ids,
            "boxes": boxes,
            "points": points,
        }
        return results
