
## RestorePadAndResize.py
class RestorePadAndResize:
    def __init__(self, scale_x=1.0, scale_y=1.0, pad_left=0, pad_top=0):
        """
        原图应该先rezsize再pad，这里还原是先去pad再还原scale
        """
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.pad_left = pad_left
        self.pad_top = pad_top

    def __call__(self, detections):
        new_detections = []
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            new_x1, new_y1, new_x2, new_y2 = (x1 - self.pad_left)/self.scale_x, (y1 - self.pad_top)/self.scale_y, \
                (x2 - self.pad_left) / self.scale_x, (y2 - self.pad_top) / self.scale_y
            box['x1'], box['y1'], box['x2'], box['y2'] = new_x1, new_y1, new_x2, new_y2
            detection['box'] = box
            # points
            points = detection["points"]
            if points is not None:
                points[:, 0] = (points[:, 0] - self.pad_left)/self.scale_x
                points[:, 1] = (points[:, 1] - self.pad_top)/self.scale_y
            new_detections.append(detection)
        return new_detections


class RestorePadAndResizeData:
    def __init__(self, process_cfg={}):
        ...

    def __call__(self, detections, img_info):
        scale_x, scale_y, pad_left, pad_top = img_info["scale_x"], img_info["scale_y"], \
            img_info["pad_left"], img_info["pad_top"]
        restore_instance = RestorePadAndResize(scale_x, scale_y, pad_left, pad_top)
        new_detections = restore_instance(detections)
        return new_detections
