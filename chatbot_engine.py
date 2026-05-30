import re
import os
from response_visualizer import ResponseVisualizer, generate_summary_chart
from screenshot_visualizer import ScreenshotVisualizer
from flask import render_template


class Chatbot:
    def __init__(self, vision_engine):
        self.engine = vision_engine
        self.detected_objects = set()  # NEW
        self.current_object = None
        self.video_ready = False
        

    def set_ready(self):
        self.video_ready = True

    def process(self, msg):
        msg = msg.lower().strip()

        if not self.video_ready:
            return "⚠️ Please upload and process the video first."

        # Trigger detection target
        if "detect" in msg:
            match = re.search(r"detect (\w+)", msg)
            if match:
                obj = match.group(1)
                self.current_object = obj
                self.detected_objects.add(self.current_object)
                return f"Now tracking: **{self.current_object}**"
            else:
                return "❓ Please specify an object to detect, e.g., 'detect car'."

        if not self.current_object:
            return "❓ Tell me what object to track first, e.g., 'detect person'."

        # Count query
        if "how many" in msg or "count" in msg:
            count = self.engine.count(self.current_object)
            return f"🔢 Total {self.current_object}s detected: **{count}**"

        # First appearance
        if "first" in msg or "appear" in msg:
            first = self.engine.first_seen(self.current_object)
            if first:
                return f"🕐 First seen at frame {first['frame_no']} ({first['timestamp']:.2f}s)"
            return f"❌ {self.current_object} not found in the video."

        # Last appearance
        if "last" in msg:
            last = self.engine.last_seen(self.current_object)
            if last:
                return f"🕓 Last seen at frame {last['frame_no']} ({last['timestamp']:.2f}s)"
            return f"❌ {self.current_object} not found in the video."

        # Positions
        if "where" in msg or "position" in msg:
            all_pos = self.engine.all_positions(self.current_object)
            if not all_pos:
                return f"❌ No {self.current_object}s found!"
            response = f"📍 Latest 3 positions of {self.current_object}:\n"
            for d in all_pos[-3:]:
                response += f"• Frame {d['frame_no']} at {d['timestamp']:.2f}s\n"
            return response

        # Screenshot visualization
        if "screenshot" in msg or "visual" in msg or "result" in msg:
            detections = self.engine.get_object_detections(self.current_object)
            if not detections:
                return f"❌ No detections found for {self.current_object}."
            
            screenshot_viz = ScreenshotVisualizer(self.engine.video_path)
            response_viz = ResponseVisualizer(self.engine.video_path)

            screenshot_viz.save_screenshots(detections, self.current_object)
            response_viz.draw_and_save(detections, self.current_object, max_frames=5)
            chart = generate_summary_chart(detections, self.current_object)

            return f"🖼️ Screenshots and summary chart saved for {self.current_object} in results folder."

        return "🤖 I didn’t understand. Try commands like: 'detect car', 'how many', 'where', 'screenshot'"


# Optional utility if needed to render template directly
def process_and_visualize_results(video_path, detections, target_class):
    visualizer = ScreenshotVisualizer(video_path)
    visualizer.save_screenshots(detections, target_class)

    screenshot_dir = "static/results"
    screenshots = [f for f in os.listdir(screenshot_dir)
                   if f.endswith(".jpg") and target_class.lower() in f.lower()]

    chart = generate_summary_chart(detections, target_class, save_dir=screenshot_dir)

    return render_template("visualization.html",
                           object_name=target_class,
                           screenshots=screenshots,
                           chart=chart)
