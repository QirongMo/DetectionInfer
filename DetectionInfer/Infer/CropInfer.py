
## CropInfer.py
from .BaseInfer import BaseInfer
from DetectionInfer.InferTools import CropImg, add_delta



class CropInfer(BaseInfer):
    def __init__(self, model_config={}):
        super().__init__(model_config)
        crop_config = model_config["Infer"].get("CropImg", {})
        self.crop = CropImg(**crop_config)
    
    def infer(self, **kwargs):
        img = kwargs["image"]
        detections = kwargs["detections"]
        img_h, img_w = img.shape[:2]
        crop_loc = self.crop(img_h, img_w, detections)
        model_detections = []
        for loc in crop_loc:
            crop_x1, crop_y1, crop_x2, crop_y2 = loc
            crop_img = img[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
            crop_detections = self.model.detect_image(crop_img)
            add_delta(crop_detections, crop_x1, crop_y1)
            model_detections += crop_detections
        return model_detections
   
