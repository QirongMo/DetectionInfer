
## TileInfer.py
from .BaseInfer import BaseInfer
from DetectionInfer.InferTools import TileImg, add_delta



class TileInfer(BaseInfer):
    def __init__(self, model_config={}):
        super().__init__(model_config)
        tile_config = model_config["Infer"]["TileImg"]
        self.tile = TileImg(**tile_config)

    def infer(self, **kwargs):
        img = kwargs["image"]
        detections = []
        img_h, img_w = img.shape[:2]
        chips = self.tile(img_h, img_w)
        for chip in chips:
            x1, y1, x2, y2 = chip
            chip_img = img[y1:y2, x1:x2]
            chip_detections = self.model_detect(chip_img)
            add_delta(chip_detections, x1, y1)
            detections += chip_detections
        return detections

