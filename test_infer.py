
import os
import cv2
from DetectionInfer import ModelInfer, InferData
from DetectionInfer.InferUtils import DrawerResult
import numpy as np


def main():
    img_path = r"samples/bus.jpg"
    img_data = InferData("")
    # img = cv2.imread(img_path)
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)

    model_yaml = "samples/yolov10.yaml"
    model_ins = ModelInfer(model_yaml)
    model_ins.load_model(0)
    model_ins.infer(img, img_data)
    detections, _ = img_data.get_results()

    drawer = DrawerResult()
    drawer.draw_detection(img, detections)
    # print(detections, "\n")
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    cv2.imwrite("test.jpg", img)


if __name__ == '__main__':
    main()
