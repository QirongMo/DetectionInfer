
import numpy as np

class DetectionBox:
    def __init__(self) -> None:
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
    
    def add_box(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
    
    def add_delta(self, x0, y0):
        if self.x1 is not None:
            self.x1 += x0
            self.x2 += x0
            self.y1 += y0
            self.y2 += y0

    def box_width(self):
        if self.x1 is None:
            return -1
        return abs(self.x2 - self.x1)
    
    def box_height(self):
        if self.y1 is None:
            return -1
        return abs(self.y2-self.y1)

    def get_dict(self):
        return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}


class DetectionResult:
    def __init__(self) -> None:
        self.box = DetectionBox()
        self.class_name = None
        self.confidence = None
        self.points: np.array = None
        self.masks: list = []
    
    def change_box(self, x1, y1, x2, y2):
        self.box.add_box(x1, y1, x2, y2)
    
    def change_confidence(self, confidence:float):
        self.confidence = confidence
    
    def change_class(self, class_name):
        self.class_name = class_name
    
    def get_dict(self):
        return {"box": self.box.get_dict(), "class_name": self.class_name, "confidence": self.confidence, 
                "points": self.points, "masks": self.masks}

    def from_dict(self, dict_data):
        box = dict_data['box']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        self.change_box(x1, y1, x2, y2)
        self.class_name = dict_data['class_name']
        self.confidence = dict_data['confidence']

    def add_delta(self, x0, y0):
        self.box.add_delta(x0, y0)
        if self.points:
            self.points[:, :0] += x0
            self.points[:, :1] += y0
        