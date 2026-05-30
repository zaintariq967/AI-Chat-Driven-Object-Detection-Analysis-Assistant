from ultralytics import YOLO

class ObjectDetectionEngine:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # Or use yolov8s.pt for more accuracy

    def detect(self, frame, conf=0.5):
        result = self.model(frame, conf=conf)[0]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.model.names[class_id]

            detections.append({
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return detections
