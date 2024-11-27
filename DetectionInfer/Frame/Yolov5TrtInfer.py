

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import os
import numpy as np
from .BaseFrame import BaseFrameInfer
from DetectionInfer.InferUtils import allocate_buffers, nms, AesCrypto


class Yolov5TrtInfer(BaseFrameInfer):
    def __init__(self, model_config={}):
        """
        :param model_config: model、label_list、thresh、nms_thresh
        """
        super().__init__(model_config)
        self.input_h, self.input_w = 0, 0
        self.error = None
        # 创建logger：日志记录器
        self.logger = None
        # 创建cuda流
        self.stream = None
        # 
        self.engine = None
        self.inputs, self.outputs, self.bindings = [], [], []
    
    def load_model(self, gpu_idx=0):
        try:
            # 创建logger：日志记录器
            self.logger = trt.Logger(trt.Logger.ERROR)
            # 创建cuda流
            self.stream = cuda.Stream()
            weight_file = self.model_config['model']
            with open(weight_file, 'rb') as f:
                weight_data = f.read()
            if self.model_config.get("encrypt"):
                self.load_network_encrypt(weight_data)
            else:
                self.load_network(weight_data)
            self.load_classes()
        except (RuntimeError, Exception) as e:
            self.error = str(e)
            
    def load_network(self, weight_data):
        with trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(weight_data) 
        inputs, outputs, bindings = allocate_buffers(engine)
        self.engine = engine
        self.inputs, self.outputs, self.bindings = inputs, outputs, bindings
        self.input_h, self.input_w = inputs[0].shape[2:]
    
    def load_network_encrypt(self, weight_data):
        aes = AesCrypto()
        weight_data = aes.decrypt_file2str(weight_data)
        self.load_network(weight_data)

    def load_classes(self):
        self.class_names += self.model_config["label_list"]
    
    def detect_image(self, img):
        if self.error is not None:
            return self.error
        try:
            return self.detect_one_image(img)
        except (RuntimeError, Exception) as e:
            self.error = str(e)
        return self.error

    def detect_one_image(self, img):
        origin_shape = img.shape[:2]
        input_img = img.transpose((2, 0, 1))
        input_img = np.ascontiguousarray(input_img)
        if len(input_img.shape) == 3:
            input_img = np.expand_dims(input_img, 0)
        engine = self.engine
        with engine.create_execution_context() as context:
            inputs, outputs, bindings = self.inputs, self.outputs, self.bindings
            inputs[0].host = input_img
            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in inputs]
            # Run inference.
            context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in outputs]
            # Synchronize the stream
            self.stream.synchronize()
            # Return only the host outputs.
            results = [out.host.reshape(out.shape) for out in outputs]
        results = self.decode_result(results[0][0])
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
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = None
        self.engine = None
        self.logger = None

