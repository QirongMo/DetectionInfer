
from math import sqrt

class LimitXY:
    def __init__(self, process_cfg={}):
        """
        :param process_cfg: dict example: class_name: {w_min_scale: 0.0, w_max_scale: 1.0}
        """
        self.process_cfg = process_cfg

    def __call__(self, detections, img_h, img_w):
        new_detections = []
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in self.process_cfg:
                new_detections.append(detection)
                continue
            process_cfg = self.process_cfg[class_name]
            box = detection['box']
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            w_scale, h_scale = abs(x2 - x1) / img_w, abs(y2 - y1) / img_h
            s_scale = sqrt(w_scale * h_scale)
            w_min_scale, w_max_scale = process_cfg.get("w_min_scale", 0), process_cfg.get("w_max_scale", 1)
            h_min_scale, h_max_scale = process_cfg.get("h_min_scale", 0), process_cfg.get("h_max_scale", 1)
            s_min_scale, s_max_scale = process_cfg.get("s_min_scale", 0), process_cfg.get("s_max_scale", 1)
            if w_min_scale < w_scale < w_max_scale and h_min_scale < h_scale < h_max_scale \
                    and s_min_scale < s_scale < s_max_scale:
                new_detections.append(detection)
        return new_detections


class LimitXYData(LimitXY):
    def __init__(self, process_cfg={}):
        super().__init__(process_cfg)

    def __call__(self, img_data):
        detections = img_data.get_results()
        img_h, img_w = img_data.img_h, img_data.img_w
        new_detections = super().__call__(detections, img_h, img_w)
        img_data.update_results(new_detections)
        return img_data
