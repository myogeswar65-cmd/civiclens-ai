import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")
video_path = "worker/test_videos/waste_test.mp4"

capture = cv2.VideoCapture(video_path)
ok, frame = capture.read()
capture.release()

print("Task:", model.task)
print("Classes:", model.names)
print("Frame loaded:", ok)

if not ok:
    raise SystemExit("Could not read the video.")

height, width = frame.shape[:2]

for threshold in [0.25, 0.50, 0.70]:
    result = model.predict(
        source=frame,
        conf=threshold,
        verbose=False,
    )[0]

    print(f"\nConfidence threshold: {threshold}")
    print("Detection count:", len(result.boxes))

    for box in result.boxes:
        x1, y1, x2, y2 = map(
            float, box.xyxy[0].tolist()
        )

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        area_percent = (
            (x2 - x1) * (y2 - y1)
            / (width * height) * 100
        )

        print(
            f"Class: {model.names[class_id]}, "
            f"confidence: {confidence:.3f}, "
            f"box: {[round(x1), round(y1), round(x2), round(y2)]}, "
            f"frame area: {area_percent:.1f}%"
        )
