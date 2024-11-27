
## PreProcess/__init__.py

from .NormalizeImage import NormalizeInput
from .Resize import ResizeInput, MaxshapeResize
from .RGBReverse import RGBReverseInput

SupportPreProcess = {
    "NormalizeInput": NormalizeInput,
    "ResizeInput": ResizeInput,
    "MaxshapeResize": MaxshapeResize,
    "RGBReverseInput": RGBReverseInput
}
