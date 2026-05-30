import os
import cv2
import matplotlib.pyplot as plt


class ResponseVisualizer:
    def __init__(self, video_path, save_dir="static/results"):
        self.video_path = video_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def draw_and_save(self, detections, object_name, max_frames=10):
        cap = cv2.VideoCapture(self.video_path)
        saved = 0

        # Use frame_no safely
        frames_to_draw = sorted({
            d.get('frame_no', d.get('frame'))
            for d in detections
            if object_name in d['class_name'].lower()
        })

        for frame_num in frames_to_draw:
            if saved >= max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Draw bounding boxes for matching objects
            for d in detections:
                if object_name in d['class_name'].lower() and d.get('frame_no', d.get('frame')) == frame_num:
                    x1, y1, x2, y2 = d.get('bbox', [0, 0, 0, 0])
                    confidence = d.get('confidence', 0.0)
                    label = f"{d['class_name']} ({confidence * 100:.1f}%)"

                    # Label background
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)

                    # Bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label text
                    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            filename = os.path.join(self.save_dir, f"{object_name}_frame_{frame_num}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[+] Saved visual frame: {filename}")
            saved += 1

        cap.release()
        return saved


def generate_summary_chart(detections, target_class, save_dir="static/results"):
    frame_numbers = [
        d.get('frame_no', d.get('frame'))
        for d in detections
        if target_class in d['class_name'].lower()
    ]

    if not frame_numbers:
        print(f"[!] No detections found for: {target_class}")
        return None

    plt.figure(figsize=(10, 4))
    plt.hist(frame_numbers, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"📊 Detection Frequency of '{target_class}'")
    plt.xlabel("Frame Number")
    plt.ylabel("Count")

    chart_filename = f"{target_class}_summary_chart.png"
    chart_path = os.path.join(save_dir, chart_filename)
    plt.savefig(chart_path)
    plt.close()

    print(f"[+] Chart saved: {chart_path}")
    return chart_filename
