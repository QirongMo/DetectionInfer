
## SupportFrame.py

from .DarknetInfer import DarknetInfer
from .Yolov5OnnxInfer import Yolov5OnnxInfer
# from .Yolov5TrtInfer import Yolov5TrtInfer
from .Yolov8OnnxInfer import Yolov8OnnxInfer
# from .Yolov8TrtInfer import Yolov8TrtInfer
from .Yolov10OnnxInfer import Yolov10OnnxInfer
# from .PaddleInfer import PaddleInfer

SupportFrame = {
    "DarknetInfer": DarknetInfer,
    "Yolov5OnnxInfer": Yolov5OnnxInfer,
    "Yolov8OnnxInfer": Yolov8OnnxInfer,
    # "Yolov5TrtInfer": Yolov5TrtInfer,
    # "Yolov8TrtInfer": Yolov8TrtInfer,
    "Yolov10OnnxInfer": Yolov10OnnxInfer,
}
