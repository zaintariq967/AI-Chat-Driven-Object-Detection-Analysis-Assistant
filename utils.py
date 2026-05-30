def save_object_screenshots(detections, video_path, object_name, save_dir):
    import cv2
    import os

    cap = cv2.VideoCapture(video_path)
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    for d in detections:
        if object_name.lower() in d['class_name'].lower():
            cap.set(cv2.CAP_PROP_POS_FRAMES, d['frame'])
            ret, frame = cap.read()
            if not ret:
                continue
            out_path = os.path.join(save_dir, f"{object_name}_{d['frame']}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append((out_path, d['timestamp']))
    cap.release()
    return saved_paths
