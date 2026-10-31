
import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")

cap = cv2.VideoCapture("worker/test_videos/waste_test.mp4")
ok, frame = cap.read()
cap.release()

print("Frame loaded:", ok)

if not ok:
    raise SystemExit("Could not read the test video.")

for size in [640, 1280]:
    result = model.predict(
        source=frame,
        conf=0.25,
        imgsz=size,
        verbose=False
    )[0]

    print(f"\nImage size: {size}")
    print("Detections:", len(result.boxes))

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        coordinates = [
            round(float(value))
            for value in box.xyxy[0].tolist()
        ]

        print(
            model.names[class_id],
            round(confidence, 3),
            coordinates
        )

    output_path = f"worker/outputs/diagnostic_{size}.jpg"
    result.save(filename=output_path)
    print("Saved:", output_path)
