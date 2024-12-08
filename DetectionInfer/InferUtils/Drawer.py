
import cv2
from math import ceil

# ttf_path = os.path.join(work_dir, "./utils/SIMHEI.TTF")
# free_type = cv2.freetype.createFreeType2()
# free_type.loadFontData(ttf_path, 0)
# line_type = 8
# font_thickness = -1


class DrawerResult:
    FREETYPE_SCALE = 0.025
    FONT_SCALE = 4e-4
    FONT_THICKNESS_SCALE = 1.0e-3
    RECTANGLE_THICKNESS = 2.5e-3

    def __init__(self):
        self.free_type = None
        self.freetype_scale = self.FREETYPE_SCALE
        self.font_scale = self.FONT_SCALE
        self.font_thickness_scale = self.FONT_THICKNESS_SCALE
        self.rectangle_thickness_scale = self.RECTANGLE_THICKNESS

    def use_freetype(self, ttf_path):
        try:
            self.free_type = cv2.freetype.createFreeType2()
            self.free_type.loadFontData(ttf_path, 0)
            return True
        except Exception as e:
            print(e)
            self.free_type = None
            return False

    def draw_detection(self, img, detections):
        """
        :param img: rgb img
        :param detections:
        :return: img
        """
        if detections is None:
            return img
        min_shape = min(img.shape[:2])
        rectangle_thickness = ceil(min_shape * self.rectangle_thickness_scale)
        if self.free_type is None:
            font_scale = min_shape * self.font_scale
            # font_face = cv2.FONT_HERSHEY_COMPLEX
            font_face = cv2.FONT_HERSHEY_SIMPLEX
        else:
            font_scale = min_shape * self.freetype_scale
        font_thickness = ceil(min_shape * self.font_thickness_scale)
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            class_name = detection['class_name']
            conf = detection['confidence']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), rectangle_thickness)
            text = str(class_name) + ' ' + str(conf)
            if self.free_type is None:
                text_size = cv2.getTextSize(text, font_face, font_scale, font_thickness)[0]
                cv2.rectangle(img, (int(x1-1), int(y1-text_size[1])), (int(x1+text_size[0]), int(y1)),
                              (0, 0, 255), -1)
                cv2.putText(img, text, (int(x1), int(y1)), font_face, font_scale, (0, 0, 0), font_thickness)
            else:
                text_size = self.free_type.getTextSize(text, font_scale, font_thickness)[0]
                cv2.rectangle(img, (x1-1, y1-text_size[1]), (x1+text_size[0], y1), (0, 0, 255), -1)
                self.free_type.putText(img, text, (x1, y1), font_scale, (0, 0, 0), font_thickness, 8, True)
            # points
            points = detection.get("points")
            if points is None:
                continue
            for point in points:
                cv2.circle(img, (int(point[0]), int(point[1])), rectangle_thickness, (0,165,255), rectangle_thickness)
        return img
