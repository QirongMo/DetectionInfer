
## InferData.py
from copy import deepcopy

class InferData:
    def __init__(self, key_name: str):
        self.key_name = key_name
        self.img_h, self.img_w = 0, 0
        self.detections = []
        self.error_detections = []

    def update_shape(self, img_h, img_w):
        self.img_h, self.img_w = img_h, img_w

    def add_result(self, detections):
        for detection in detections:
            if isinstance(detection, dict):
                self.detections.append(deepcopy(detection))
            else:
                self.error_detections.append(deepcopy(detection))

    def update_results(self, detections):
        self.detections.clear()
        self.error_detections.clear()
        self.add_result(detections)

    def get_results(self):
        return self.detections, self.error_detections

    def __str__(self):
        return f"InferData({self.key_name})"

