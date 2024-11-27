
## ModelInfer.py

import os
import sys
dir_name = os.path.dirname(__file__)
if dir_name not in sys.path:
    sys.path.append(dir_name)
parent_path = os.path.abspath(os.path.join(dir_name, '../'))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import yaml

from DetectionInfer.Infer import SupportInfer
from DetectionInfer.InferProcess import SupportInferProcess



class ModelInfer:
    def __init__(self, model_yaml):
        self.model_yaml = model_yaml
        self.model_cfg = None
        self.model_infer = None

    def load_model(self, gpu_idx=0,):
        with open(self.model_yaml, "r") as f:
            model_cfg = yaml.safe_load(f)
        self.model_cfg = model_cfg
        infer_cfg = model_cfg["Infer"]
        infer_type = infer_cfg["type"]
        # model_infer = getattr(Infer, infer_type)(model_cfg)
        model_infer = SupportInfer[infer_type](model_cfg)
        model_infer.load_model(gpu_idx)
        self.model_infer = model_infer

    def infer(self, img, infer_data):
        result, _ = infer_data.get_results()
        detections = self.model_infer.infer(image=img, detections=result)
        if isinstance(detections, list):
            infer_data.add_result(detections)
            self.reprocess(infer_data)
        return detections

    def reprocess(self, infer_data):
        infer_process = self.model_cfg.get("InferProcess", [])
        for p in infer_process:
            for k, v in p.items():
                # op_cls = getattr(InferProcess, k)
                op_cls = SupportInferProcess[k]
                f = op_cls(v)
                f(infer_data)

    def clean_model(self):
        self.model_infer.clean_model()
        self.model_infer = None
