
## BaseInfer.py
from DetectionInfer.Frame import FrameInfer


class BaseInfer:
    def __init__(self, model_config={}):
        self.model_config = model_config
        self.model = FrameInfer(model_config)

    def load_model(self, gpu_idx=0):
        self.model.load_model(gpu_idx)

    def infer(self, **kwargs):
        # 添加results参数是因为有些操作要基于前面的识别结果才能进行
        img = kwargs["image"]
        detections = self.model_detect(img)
        return detections

    def model_detect(self, img):
        detections = self.model.detect_image(img)
        return detections

    def clean_model(self):
        self.model.clean_model()
