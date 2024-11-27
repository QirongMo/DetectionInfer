
## TileImg
class TileImg:
    def __init__(self, clip_size, stride_size):
        self.clip_size = clip_size
        self.stride_size = stride_size

    def __call__(self, img_h, img_w):
        chips = []
        stride = int(self.clip_size * self.stride_size)
        for i in range(0, img_w - self.clip_size, stride):
            for j in range(0, img_h - self.clip_size, stride):
                x1 = i
                y1 = j
                x2 = i + self.clip_size - 1
                y2 = j + self.clip_size - 1
                chips.append([x1, y1, x2, y2])
        # 下边缘
        for i in range(0, img_w - self.clip_size, stride):
            y1 = max(img_h - self.clip_size, 0)
            y2 = img_h - 1
            x1 = i
            x2 = i + self.clip_size - 1
            chips.append([x1, y1, x2, y2])
        # 右边缘
        for j in range(0, img_h - self.clip_size, stride):
            x1 = max(img_w - self.clip_size, 0)
            x2 = img_w - 1
            y1 = j
            y2 = j + self.clip_size - 1
            chips.append([x1, y1, x2, y2])
        # 右下角
        chips.append([max(img_w - self.clip_size, 0), max(img_h - self.clip_size, 0), img_w - 1, img_h - 1])
        return chips


def add_delta(chip_detections, x1, y1):
    for detection in chip_detections:
        box = detection["box"]
        box['x1'] += x1
        box['y1'] += y1
        box['x2'] += x1
        box['y2'] += y1
    return chip_detections
