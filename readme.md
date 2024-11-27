
# DectctionInfer

python多框架目标检测模型推理，目前支持[alexeyab的darknet框架](https://github.com/AlexeyAB/darknet)的yolov3和yolov4、[ultralytics的yolov5](https://github.com/ultralytics/yolov5.git)、[ultralytics的ultralytics库](https://github.com/ultralytics/ultralytics)的yolov8和yolov10。其中yolov5、yolov8、yolov10都是使用onnx推理，而yolov5、yolov8还支持tensorrt推理。

1、安装
```bash
pip install -e .
```

2、运行推理例子
```bash
python test_infer.py
```