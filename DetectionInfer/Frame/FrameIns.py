
## FrameIns.py
import os
from .SupportFrame import SupportFrame
from DetectionInfer.PreProcess import SupportPreProcess
from DetectionInfer.ReProcess import SupportReProcess


def init_img_info():
    return {
        "scale_x": 1.0, 
        "scale_y": 1.0, 
        "pad_left": 0, 
        "pad_top": 0
    }

class FrameInfer:
    def __init__(self, model_cfg={}):
        self.model_cfg = model_cfg
        self.model = None
        # cuda dll path
        dll_path = os.path.abspath(os.path.join(os.getcwd(), "./dll/cuda"))
        os.environ['PATH'] = dll_path + ';' + os.environ['PATH']

    def load_model(self, gpu_idx=0):
        if self.model is not None:
            return False
        frame_data = self.model_cfg["FrameInfer"]
        frame_type = frame_data["type"]
        frame_model = SupportFrame[frame_type]
        self.model = frame_model(frame_data)
        self.model.load_model(gpu_idx)

    def preprocess(self, img, img_info):
        preprocess = self.model_cfg.get("PreProcess", [])
        for p in preprocess:
            for k, v in p.items():
                # op_cls = getattr(PreProcess, k)
                op_cls = SupportPreProcess[k]
                f = op_cls(**v)
                img, img_info = f(img, img_info)
        return img, img_info

    def reprocess(self, detections, img_info):
        reprocess = self.model_cfg.get("ReProcess", [])
        for p in reprocess:
            for k, v in p.items():
                # op_cls = getattr(ReProcess, k)
                op_cls = SupportReProcess[k]
                f = op_cls(v)
                detections = f(detections, img_info)
        return detections       

    def detect_image(self, img):
        img_info = init_img_info()
        img, img_info = self.preprocess(img, img_info)
        detections = self.model.detect_image(img)
        if isinstance(detections, list):
            detections = self.reprocess(detections, img_info)
        return detections

    def clean_model(self):
        self.model.clean_model()
        self.model = None
