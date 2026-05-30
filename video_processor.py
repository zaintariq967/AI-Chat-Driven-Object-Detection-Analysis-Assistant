import os
import cv2

class VideoProcessor:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps == 0:
            raise ValueError("FPS is zero. Invalid or corrupt video file.")

    def extract_frames(self, interval=1):
        count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if count % interval == 0:
                yield frame, count, count / self.fps

            count += 1
            if count % 50 == 0:
                print(f"[VideoProcessor] Processed {count}/{self.frames} frames")

    def release(self):
        self.cap.release()
