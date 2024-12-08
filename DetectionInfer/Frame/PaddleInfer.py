
from .BaseFrame import BaseFrameInfer
import paddle.inference as paddle_infer
import numpy as np


class PaddleModel(BaseFrameInfer):
    def __init__(self, model_config={}):
        """
        :param model_config: cfg、weight、thresh、nms_thresh
        """
        super().__init__(model_config)

    def load_config(self, gpu_idx=0):
        infer_model = self.model_config['infer_model']  # 'model.pdmodel'
        infer_params = self.model_config['infer_params']  # model.pdiparams
        config = paddle_infer.Config(infer_model, infer_params)
        device = "cpu"
        if device.lower() == 'gpu':
            # initial GPU memory(M), device ID
            config.enable_use_gpu(200, gpu_idx)
            # optimize graph and fuse op
            # config.switch_ir_optim(True)
        else:
            cpu_threads = 1
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_threads)
        # disable print log when predict
        config.disable_glog_info()
        # enable shared memory
        config.enable_memory_optim()
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        return config

    def load_model(self, gpu_idx=0):
        config = self.load_config()
        # 根据 config 创建 predictor
        self.network = paddle_infer.create_predictor(config)
        self.load_classes()

    def load_classes(self):
        self.class_names += self.model_config["label_list"]

    def detect_image(self, img):
        input_h, input_w = img.shape[:2]
        img_info = {
            'im_shape': [input_h, input_w],
            "scale_factor": [1.0, 1.0]  # [scale_y, scale_x]
        }
        self.input_img(img, img_info)
        #  predict
        self.network.run()
        output_names = self.network.get_output_names()
        boxes_tensor = self.network.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        thresh = self.model_config.get('thresh', 0.1)
        expect_boxes = (np_boxes[:, 1] > thresh) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        detections = self.decode_detection(np_boxes)
        return detections

    def decode_detection(self, np_boxes):
        predictions = []
        for box in np_boxes:
            class_id, conf, x1, y1, x2, y2 = box.tolist()
            box = {'x1': round(x1, 2), 'y1': round(y1, 2), 'x2': round(x2, 2), 'y2': round(y2, 2)}
            class_name = self.class_names[int(class_id)]
            result = {'box': box, 'class_name': class_name, 'confidence': conf}
            predictions.append(result)
        return predictions

    def input_img(self, img, img_info):
        # 输入
        img = img.transpose((2, 0, 1))
        inputs = {
            'image': np.array((img,)).astype('float32'),
            'im_shape': np.array((img_info['im_shape'],)).astype('float32'),
            'scale_factor': np.array((img_info['scale_factor'],)).astype('float32')
        }
        input_names = self.network.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.network.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

    def clean_model(self):
        if self.network is not None:
            self.network = None
            self.class_names.clear()


class PaddleInfer(PaddleModel, BaseFrameInfer):
    def __init__(self, model_config={}):
        super().__init__(model_config)

    def detect(self, input_data):
        img = input_data.img
        input_h, input_w = img.shape[:2]
        # img_info = {
        #     'im_shape': [input_h, input_w],
        #     "scale_factor": [input_data.scale_y, input_data.scale_x]
        # }
        img_info = {
            'im_shape': [input_h, input_w],
            "scale_factor": [1.0, 1.0]
        }
        detections = self.detect_image(img, img_info)
        input_data.add_detections(detections)
