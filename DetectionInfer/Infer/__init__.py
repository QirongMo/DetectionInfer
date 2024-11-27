
## Infer/__init__.py

from .BaseInfer import BaseInfer
from .TileInfer import TileInfer
from .CropInfer import CropInfer

SupportInfer= {
    "BaseInfer": BaseInfer,
    "TileInfer": TileInfer,
    "CropInfer": CropInfer
}
