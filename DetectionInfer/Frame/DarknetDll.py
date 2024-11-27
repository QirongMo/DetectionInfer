
## DarknetDll,py
from ctypes import *
import os

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("best_class_idx", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def dbox2points(bbox):
    return bbox2points([bbox.x, bbox.y, bbox.w, bbox.h])


def decode_detection(detections, class_names, num):
    predictions = []
    for j in range(num):
        detection = detections[j]
        for idx, name in enumerate(class_names):
            if detection.prob[idx] > 0:
                x1, y1, x2, y2 = dbox2points(detection.bbox)
                box = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                conf = detection.prob[idx]
                result = {'box': box, 'class_name': name, 'confidence': conf}
                predictions.append(result)
    return predictions


def decode_detection_faster(detections, class_names, num):
    """
    Faster version of remove_negatives (very useful when using yolo9000)
    """
    predictions = []
    for j in range(num):
        detection = detections[j]
        class_id = detection.best_class_idx
        if class_id == -1:
            continue
        name = class_names[class_id]
        x1, y1, x2, y2 = dbox2points(detection.bbox)
        box = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        conf = detection.prob[class_id]
        result = {'box': box, 'class_name': name, 'confidence': conf}
        predictions.append(result)
    return predictions


class DarknetDll:
    def __init__(self, dll_path):
        self.lib = None
        self.set_gpu = None
        self.load_net_custom = None
        self.load_net_custom_encry = None
        self.make_image = None
        self.copy_image_from_bytes = None
        self.predict_image = None
        self.get_network_boxes = None
        self.do_nms_sort = None
        self.free_detections = None
        self.free_image = None
        self.free_network_ptr = None
        self.load_lib(dll_path)

    def load_lib(self, dll_path):
        os.environ['PATH'] = dll_path + ';' + os.environ['PATH']
        if os.name == "posix":
            self.lib = CDLL(os.path.join(dll_path, "libdarknet.so"), RTLD_GLOBAL)
        elif os.name == "nt":
            self.lib = CDLL(os.path.join(dll_path, "yolo_cpp_dll_gpu.dll"), RTLD_GLOBAL, winmode=0x8)
        else:
            print("Unsupported OS")
    
    def select_gpu(self, gpu_idx=0):
        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]
        self.set_gpu(gpu_idx)
    
    def load_network(self, config_file, weights, batch_size = 1, encrypt=False):
        """
        load model description and weights from config files
        args:
            config_file (str): path to .cfg model file
            weights (str): path to weights
        returns:
            network: trained model
        """
        if encrypt:
            return self.load_net_encrypt(config_file, weights, batch_size)
        else:
            return self.load_net(config_file, weights, batch_size)

    def load_net(self, config_file, weights, batch_size = 1):
        if self.load_net_custom is None:
            self.load_net_custom = self.lib.load_network_custom
            self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
            self.load_net_custom.restype = c_void_p
        network = self.load_net_custom(
            config_file,
            weights, 0, batch_size)
        return network

    def load_net_encrypt(self, config_file, weights, batch_size = 1):
        if self.load_net_custom_encry is None:
            self.load_net_custom_encry = self.lib.load_network_custom_encry
            self.load_net_custom_encry.argtypes = [c_char_p, c_char_p, c_int, c_int, c_int]
            self.load_net_custom_encry.restype = c_void_p
        network = self.load_net_custom_encry(
            config_file,
            weights, 1, batch_size, 2023)
        return network

    def img2darknet(self, img):
        if self.make_image is None:
            self.make_image = self.lib.make_image
            self.make_image.argtypes = [c_int, c_int, c_int]
            self.make_image.restype = IMAGE
            self.copy_image_from_bytes = self.lib.copy_image_from_bytes
            self.copy_image_from_bytes.argtypes = [IMAGE,c_char_p]
        img_h, img_w = img.shape[:2]
        darknet_image = self.make_image(img_w, img_h, 3)
        self.copy_image_from_bytes(darknet_image, img.tobytes())
        return darknet_image
    
    def detect_image(self, network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
        """
            Returns a list with highest confidence class and their bbox
        """
        if self.predict_image is None:
            self.predict_image = self.lib.network_predict_image
            self.predict_image.argtypes = [c_void_p, IMAGE]
            self.predict_image.restype = POINTER(c_float)
            # 
            self.get_network_boxes = self.lib.get_network_boxes
            self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
            self.get_network_boxes.restype = POINTER(DETECTION)
            # 
            self.do_nms_sort = self.lib.do_nms_sort
            self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
            # 
            self.free_detections = self.lib.free_detections
            self.free_detections.argtypes = [POINTER(DETECTION), c_int]
        pnum = pointer(c_int(0))
        self.predict_image(network, image)
        detections = self.get_network_boxes(network, image.w, image.h,
                                    thresh, hier_thresh, None, 0, pnum, 0)
        num = pnum[0]
        if nms:
            self.do_nms_sort(detections, num, len(class_names), nms)
        predictions = decode_detection(detections, class_names, num)
        self.free_detections(detections, num)
        return sorted(predictions, key=lambda x: x['confidence'])
    
    def free_darknet_image(self, darknet_image):
        if self.free_image is None:
            self.free_image = self.lib.free_image
            self.free_image.argtypes = [IMAGE]   
        self.free_image(darknet_image)
    
    def free_network(self, network):
        if self.free_network_ptr is None:
            self.free_network_ptr = self.lib.free_network_ptr
            self.free_network_ptr.argtypes = [c_void_p]
            self.free_network_ptr.restype = c_void_p
        self.free_network_ptr(network)