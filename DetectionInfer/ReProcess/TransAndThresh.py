
## TransAndThresh.py
class TransAndThresh:
    def __init__(self, process_data={}):
        """
        :param process_data: dict example: class_name: {trans_name: new_name, thresh: 0.25}
        """
        self.process_data = process_data

    def __call__(self, detections):
        new_detections = []
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in self.process_data:
                new_detections.append(detection)
                continue
            process_data = self.process_data[class_name]
            new_name = process_data.get("trans_name", class_name)
            detection['class_name'] = new_name
            confidence = detection['confidence']
            thresh = process_data.get("thresh", -1)
            if confidence <= thresh:
                continue
            new_detections.append(detection)
        return new_detections


class TransAndThreshData(TransAndThresh):
    def __init__(self, process_data={}):
        super().__init__(process_data)

    def __call__(self, detections, img_info):
        new_detections = super().__call__(detections)
        return new_detections
