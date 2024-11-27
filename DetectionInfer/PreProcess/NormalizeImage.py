
## NormalizeImage.py
import numpy as np


class NormalizeImage:
    def __init__(self, mean=0, std=1, is_scale=True):
        self.mean = mean if isinstance(mean, list) else [mean, mean, mean]
        self.std = std if isinstance(std, list) else [std, std, std]
        self.is_scale = is_scale
        
    def __call__(self, img):
        img = img.astype(np.float32)
        if self.is_scale:
            img /= 255.0
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        img -= mean
        img /= std
        return img


class NormalizeInput(NormalizeImage):
    def __init__(self, mean=0, std=1, is_scale=True):
        super().__init__(mean, std, is_scale)

    def __call__(self, img, img_info):
        img = super().__call__(img)
        return img, img_info
