from video_processor import VideoProcessor
from detection_engine import ObjectDetectionEngine
from vision_query_engine import VisionQueryEngine
from chatbot_engine import Chatbot

def process_video(video_path, vision_engine):
    vp = VideoProcessor(video_path)
    detector = ObjectDetectionEngine()

    print("Processing video...")
    frame_counter = 0

    for frame, frame_no, ts in vp.extract_frames():
        detections = detector.detect(frame)
        vision_engine.log_detections(frame_no, ts, detections)
        frame_counter += 1
        if frame_counter % 30 == 0:
            print(f"Processed {frame_counter} frames...")

    vp.release()
    print("Video processing complete.")

def start_chat_loop(chatbot):
    print("\nChatbot Ready! Type your queries below:")
    print("Examples: 'detect car', 'how many?', 'when?', 'where?', 'last?', 'exit'\n")

    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ['exit', 'quit']:
            print("Exiting. Shukriya!")
            break
        reply = chatbot.process(user_msg)
        print(f"Bot: {reply}")

def main():
    print(" AI Object Detection CLI Assistant")
    video_path = input("Enter video path (e.g., traffic-mini.mp4): ").strip()

    # Initialize components
    vision_engine = VisionQueryEngine(video_path)
    chatbot = Chatbot(vision_engine)

    # Step 1: Process Video
    process_video(video_path, vision_engine)

    # Step 2: Activate chatbot
    chatbot.set_ready()
    start_chat_loop(chatbot)

if __name__ == "__main__":
    main()
