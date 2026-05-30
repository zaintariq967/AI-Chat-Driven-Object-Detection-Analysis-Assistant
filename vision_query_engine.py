class VisionQueryEngine:
    def __init__(self, video_path):
        self.detection_history = []
        self.video_path = video_path  # Store path for screenshot visualizer


    # vision_query_engine.py
    def log_detections(self, frame_num, timestamp, detections):
        for d in detections:
            self.detection_history.append({
            "class_name": d["class_name"],
            "frame_no": frame_num,   #  use "frame_no" to match ScreenshotVisualizer
            "timestamp": timestamp,
            "confidence": d["confidence"],
            "bbox": d["bbox"]        #  include bbox for visualization!
        })


    def count(self, object_name):
        return sum(1 for d in self.detection_history if object_name in d["class_name"].lower())

    def first_seen(self, object_name):
        for d in self.detection_history:
            if object_name in d["class_name"].lower():
                return d
        return None

    def last_seen(self, object_name):
        for d in reversed(self.detection_history):
            if object_name in d["class_name"].lower():
                return d
        return None

    def all_positions(self, object_name):
        return [d for d in self.detection_history if object_name in d["class_name"].lower()]
 
    def get_object_detections(self, object_name):
        return [d for d in self.detection_history if object_name in d["class_name"].lower()]
