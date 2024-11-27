

class BaseFrameInfer:
    def __init__(self, model_config={}):
        self.model_config = model_config
        self.network = None
        self.class_names = []

    def load_model(self, gpu_idx=0): ...
    
    def load_classes(self): ...
    
    def detect_image(self, img):
        detections = []
        return detections
    
    def clean_model(self): ...

