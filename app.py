from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from threading import Thread

from chatbot_engine import Chatbot
from detection_engine import ObjectDetectionEngine
from video_processor import VideoProcessor
from vision_query_engine import VisionQueryEngine
from screenshot_visualizer import ScreenshotVisualizer
from response_visualizer import generate_summary_chart, ResponseVisualizer

processing_status = {"done": False}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

chatbot = query_engine = video_path = None  # Global vars

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/process_video', methods=['POST'])
def process_video():
    global chatbot, query_engine, video_path, processing_status

    file = request.files['video']
    frame_step = int(request.form.get("frame_step", 5))  # From frontend dropdown
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    # Reset status before starting background processing
    processing_status["done"] = False

    def background_task():
        global chatbot, query_engine, processing_status

        vp = VideoProcessor(video_path)
        detector = ObjectDetectionEngine()
        query_engine = VisionQueryEngine(video_path)

        processed, skipped = 0, 0
        for frame, frame_no, ts in vp.extract_frames():
            if frame_no % frame_step != 0:
                skipped += 1
                continue
            query_engine.log_detections(frame_no, ts, detector.detect(frame))
            processed += 1

        vp.release()
        chatbot = Chatbot(query_engine)
        chatbot.set_ready()

        processing_status["done"] = True
        print(f"[INFO] Finished: {processed} frames processed, {skipped} skipped.")

    Thread(target=background_task).start()
    return jsonify({"message": "✅ Video is being processed in background. Please wait a moment before asking."})




@app.route('/chat', methods=['POST'])
def chat():
    global chatbot
    msg = request.json['message']
    reply = chatbot.process(msg) if chatbot else "⚠️ Video not yet processed. Please upload a video first."
    return jsonify({"reply": reply})

@app.route('/visualize_all')
def visualize_all():
    global chatbot, video_path
    if not chatbot or not video_path:
        return "Video not processed yet."

    engine = chatbot.engine
    detected_objects = chatbot.detected_objects  # already tracked by chatbot

    vis1 = ResponseVisualizer(video_path, save_dir=app.config['RESULT_FOLDER'])
    vis2 = ScreenshotVisualizer(video_path, save_dir=app.config['RESULT_FOLDER'])

    max_screenshots = request.args.get("limit", 5, type=int)

    charts = []
    screenshots_by_object = {}

    for obj in detected_objects:
        obj = obj.lower()

        # Generate chart and screenshots
        vis1.draw_and_save(engine.detection_history, obj)
        vis2.save_screenshots(engine.detection_history, obj, max_screenshots=max_screenshots)

        chart = generate_summary_chart(engine.detection_history, obj, save_dir=app.config['RESULT_FOLDER'])
        if chart:
            charts.append((obj, chart))

        screenshots = [
            f for f in os.listdir(app.config['RESULT_FOLDER'])
            if obj in f and f.endswith(".jpg")
        ]
        screenshots_by_object[obj] = screenshots

    return render_template("visualization_multi.html", charts=charts, screenshots=screenshots_by_object)



@app.route('/static/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/processing_status')
def check_processing_status():
    return jsonify({"done": processing_status["done"]})


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
