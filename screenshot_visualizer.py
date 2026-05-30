import cv2
import os
import numpy as np

class ScreenshotVisualizer:
    def __init__(self, video_path, save_dir="static/results"):
        self.video_path = video_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_screenshots(self, detection_history, target_class, max_screenshots=5):
        cap = cv2.VideoCapture(self.video_path)

        # ✅ Filter for the correct object and valid frame numbers
        filtered = [
            d for d in detection_history
            if d.get('class_name', '').lower() == target_class.lower() and
               (d.get('frame_no') is not None or d.get('frame') is not None)
        ]

        if not filtered:
            return

        # ✅ Sort safely with fallback
        filtered.sort(key=lambda d: d.get('frame_no') or d.get('frame') or -1)

        # ✅ Get unique frame numbers
        unique_frames = sorted(set(d.get('frame_no') or d.get('frame') for d in filtered if (d.get('frame_no') or d.get('frame')) is not None))
        total = len(unique_frames)
        if total == 0:
            return

        # ✅ Spread-out sampling
        step = max(1, total // max_screenshots)
        selected_frames = [unique_frames[i] for i in range(0, total, step)][:max_screenshots]

        # ✅ Save each screenshot
        for frame_no in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue

            # Use first detection for this frame
            detection = next((d for d in filtered if (d.get('frame_no') or d.get('frame')) == frame_no), None)
            if not detection:
                continue

            # Draw bounding box if available
            x1, y1, x2, y2 = detection.get('bbox', [10, 10, 100, 100])
            label = f"{target_class} ({detection.get('confidence', 0):.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            filename = os.path.join(self.save_dir, f"{target_class}_frame_{frame_no}.jpg")
            cv2.imwrite(filename, frame)

        cap.release()
