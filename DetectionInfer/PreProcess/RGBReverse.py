
## RGBReverse.py
import numpy as np


class RGBReverse:
    def __init__(self):
        ...

    def __call__(self, img):
        return np.ascontiguousarray(img[:, :, ::-1])


class RGBReverseInput(RGBReverse):
    def __init__(self):
        ...

    def __call__(self, img, img_info):
        img = super().__call__(img)
        return img, img_info
