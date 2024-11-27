
## Yolov5OnnxInfer.py
import onnxruntime
import numpy as np
from .BaseFrame import BaseFrameInfer
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException, NoSuchFile, Fail, InvalidArgument
from DetectionInfer.InferUtils import nms, AesCrypto


class Yolov5OnnxInfer(BaseFrameInfer):
    def __init__(self, model_config={}):
        """
        :param model_config: model、label_list、thresh、nms_thresh
        """
        super().__init__(model_config)
        self.input_h, self.input_w = 0, 0
        self.error = None
    
    def load_model(self, gpu_idx=0):
        try:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': gpu_idx,
                    # 'gpu_mem_limit': 3 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'HEURISTIC', # 注意这里，搜索算法可能会造成使用大量显存
                }),
                'CPUExecutionProvider',
            ]
            encrypt = self.model_config.get('encrypt', False)
            if encrypt:
                self.load_network_encrypt(providers)
            else:
                self.load_network(providers)
            self.input_h, self.input_w = self.network.get_inputs()[0].shape[2:4]
            self.load_classes()
        except (RuntimeException, Fail, NoSuchFile) as e:
            self.error = str(e)
        except (RuntimeError, Exception) as e:
            self.error = str(e)

    def load_network(self, providers):
        onnx_file = self.model_config['model']  # 'model.onnx'
        self.network = onnxruntime.InferenceSession(onnx_file, providers=providers)  
    
    def load_network_encrypt(self, providers):
        onnx_file = self.model_config['model']  # 'model.onnx'
        aes = AesCrypto()
        onnx_byte = aes.decrypt_file2str(onnx_file)
        self.network = onnxruntime.InferenceSession(onnx_byte, providers=providers)
    
    def warm_up(self):
        # Warmup model by running inference once
        img = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
        self.network.run([self.network.get_outputs()[0].name],
                         {self.network.get_inputs()[0].name: img})

    def load_classes(self):
        self.class_names += self.model_config["label_list"]
    
    def detect_image(self, img):
        if self.error is not None:
            return self.error
        try:
            return self.detect_one_image(img)
        except InvalidArgument as e:
            self.error = str(e)
        except (RuntimeError, Exception) as e:
            self.error = str(e)
        if self.error:
            print(self.error)
        return self.error

    def detect_one_image(self, img):
        origin_shape = img.shape[:2]
        img = img.transpose((2, 0, 1))
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        pred_results = self.network.run([self.network.get_outputs()[0].name],
                                        {self.network.get_inputs()[0].name: img})[0]
        results = self.decode_result(pred_results[0])
        conf_thresh, iou_thresh = self.model_config.get('thresh', 0.1), self.model_config.get('nms_thresh', 0.35)
        results = nms(results, conf_thresh, iou_thresh)
        detections = self.decode_detection(results, origin_shape)
        return detections

    def decode_result(self, pred_results):
        boxes = pred_results[..., :4]
        obj_conf = pred_results[..., 4:5]
        anchor_conf = pred_results[..., 5:5+len(self.class_names)]
        anchor_max_conf = np.max(anchor_conf, axis=1, keepdims=True)
        class_ids = np.argmax(anchor_conf, axis=1, keepdims=True)
        class_conf = anchor_max_conf*obj_conf
        results = {
            "confs": class_conf,
            "class_ids": class_ids,
            "boxes": boxes,
            "points": None,
        }
        return results
    
    def decode_detection(self, results, origin_shape):
        predictions = []
        confs, class_ids, boxes, points = results["confs"], results["class_ids"], \
            results["boxes"], results.get("points")
        for i in range(len(boxes)):
            x1 = float(boxes[i, 0]*origin_shape[1] / self.input_w)
            y1 = float(boxes[i, 1]*origin_shape[0] / self.input_h)
            x2 = float(boxes[i, 2]*origin_shape[1] / self.input_w)
            y2 = float(boxes[i, 3]*origin_shape[0] / self.input_h)
            box = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            # class name
            class_id = class_ids[i]
            class_name = self.class_names[int(class_id)]
            # conf
            conf = round(float(confs[i]), 4)
            # points
            _points = points[i] if points is not None else None
            num_points = self.model_config.get("num_points")
            if _points is not None:
                num_points = _points.shape[0]//3 if num_points is None else num_points
                _points = _points.reshape((num_points, -1))
                _points[i, 0] *= origin_shape[1] / self.input_w
                _points[i, 1] *= origin_shape[0] / self.input_h
            result = {'box': box, 'class_name': class_name, 'confidence': conf, "points": _points}
            predictions.append(result)
        return predictions
    
    def clean_model(self):
        if self.network is not None:
            self.network = None
            self.class_names.clear()
           
