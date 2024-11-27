
## DarknetInfer.py
import os
from .BaseFrame import BaseFrameInfer
from .DarknetDll import DarknetDll
from DetectionInfer.InferUtils import AesCrypto


class DarknetInfer(BaseFrameInfer):
    def __init__(self, model_config={}):
        """
        :param model_config: cfg、weight、thresh、nms_thresh
        """
        super().__init__(model_config)
        # self.cfx = cuda0.Device(0).make_context()
        # dll_path = os.path.abspath(os.path.join(os.getcwd(), "./dll/darknet"))
        dll_path = os.environ['DarknetPath']
        self.darknet = DarknetDll(dll_path)

    def load_model(self, gpu_idx=0):
        cfg_file = self.model_config['cfg']
        weight_file = self.model_config['weight']
        self.darknet.select_gpu(gpu_idx)
        encrypt = self.model_config.get('encrypt', False)
        if encrypt:
            aes = AesCrypto()
            cfg = aes.decrypt_file2str(cfg_file)
        else:
            cfg = cfg_file.encode("ascii")
        self.network = self.darknet.load_network(cfg, weight_file.encode("ascii"), encrypt=encrypt)
        self.load_classes()

    def load_classes(self):
        self.class_names += self.model_config["label_list"]
        # name_file = self.model_config['names']
        # class_names = []
        # with open(name_file, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         class_name = line.strip()
        #         class_names.append(class_name)
        # self.class_names += class_names

    def detect_image(self, img):
        # self.cfx.push()
        thresh = self.model_config.get('thresh', 0.1)
        nms_thresh = self.model_config.get('nms_thresh')
        darknet_image = self.darknet.img2darknet(img)
        detections = self.darknet.detect_image(self.network, self.class_names, darknet_image, thresh=thresh, nms=nms_thresh)
        self.darknet.free_darknet_image(darknet_image)
        return detections
    
    def clean_model(self):
        if self.network is not None:
            self.darknet.free_network(self.network)
            self.network = None
            self.class_names.clear()
            # self.context.execute_v2(list(self.binding_addrs.values()))
            # self.cfx.pop()

