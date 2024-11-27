
## Resize.py
import cv2


class Resize:
    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR, pad_color=114):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp
        if isinstance(pad_color, int):
            pad_color = [pad_color, pad_color, pad_color]
        self.pad_color = pad_color

    def __call__(self, img):
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        org_h, org_w = img.shape[:2]
        scale_x, scale_y = self.generate_scale(org_h, org_w)
        scale_size = {
            "scale_x": scale_x,
            "scale_y": scale_y
        }
        img = cv2.resize(img, (0, 0),
                         fx=scale_x,
                         fy=scale_y,
                         interpolation=self.interp)
        new_h, new_w = img.shape[:2]
        target_w, target_h = self.target_size
        pad_left, pad_top = (target_w - new_w) // 2, (target_h - new_h) // 2
        pad_right, pad_bottom = target_w - new_w - pad_left, target_h - new_h - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                 value=self.pad_color)
        pad_size = {
            "pad_left": pad_left,
            "pad_top": pad_top
        }
        return img, scale_size, pad_size

    def generate_scale(self, org_h, org_w):
        target_w, target_h = self.target_size
        if self.keep_ratio:
            k = min(target_w / org_w, target_h / org_h)
            scale_x, scale_y = k, k
        else:
            scale_x, scale_y = target_w / org_w, target_h / org_h
        return scale_x, scale_y


class ResizeInput(Resize):
    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        super().__init__(target_size, keep_ratio, interp)

    def __call__(self, img, img_info):
        img, scale_size, pad_size = super().__call__(img)
        img_info["scale_x"] *= scale_size["scale_x"]
        img_info["scale_y"] *= scale_size["scale_y"]
        if not (img_info["pad_left"] or img_info["pad_top"]):
            img_info["pad_left"], img_info["pad_top"] = pad_size["pad_left"], pad_size["pad_top"]
        return img, img_info


class MaxshapeResize(Resize):
    def __init__(self, max_shape, interp=cv2.INTER_LINEAR):
        self.max_shape = max_shape
        self.interp = interp

    def __call__(self, img, img_info):
        k = self.max_shape/max(img.shape[:2])
        img = cv2.resize(img, (0, 0),
                         fx=k,
                         fy=k,
                         interpolation=self.interp)
        img_info["scale_x"] *= k
        img_info["scale_y"] *= k
        return img, img_info
