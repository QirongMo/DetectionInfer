
## CropImg.py
class CropImg:
    def __init__(self, alias_class=None, rect=False, ratio=1.0):
        self.alias_class: list = alias_class
        self.rect = rect
        self.ratio = ratio

    def __call__(self, img_h, img_w, detections):
        crop_loc = []
        for detection in detections:
            if self.alias_class is not None and detection['class_name'] not in self.alias_class:
                continue
            box = detection['box']
            box_x1, box_y1, box_x2, box_y2 = box['x1'], box['y1'], box['x2'], box['y2']
            c_x, c_y = (box_x1+box_x2)/2.0, (box_y1+box_y2)/2.0
            box_w, box_h = abs(box_x2-box_x1), abs(box_y2-box_y1)
            if self.rect:
                crop_w, crop_h = box_w*self.ratio, box_h*self.ratio
            else:
                crop_shape = max(box_w, box_h)*self.ratio
                crop_w, crop_h = crop_shape, crop_shape
            crop_x1, crop_y1 = max(0, c_x-crop_w/2.0), max(0, c_y-crop_h/2.0)
            crop_x2, crop_y2 = min(c_x+crop_w/2.0, img_w), min(c_y+crop_h/2.0, img_h)
            # crop_x1, crop_y1, crop_x2, crop_y2 = int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)
            crop_loc.append([crop_x1, crop_y1, crop_x2, crop_y2])
        return crop_loc

