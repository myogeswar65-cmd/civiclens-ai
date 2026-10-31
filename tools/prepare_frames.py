
from pathlib import Path
import cv2
import re

BASE = Path(__file__).resolve().parent
VIDEO_DIR = BASE / "videos"

# Separate folder so the previous extraction stays safe
OUTPUT_DIR = (
    BASE / "datasets" / "bin_condition"
    / "images" / "raw_frames_v2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_SECONDS = 2

videos = sorted(VIDEO_DIR.glob("*.mp4"))
total_saved = 0

for video_index, video_path in enumerate(videos, start=1):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print("Could not open:", video_path.name)
        continue

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        continue

    safe_name = re.sub(
        r"[^a-zA-Z0-9]+", "_", video_path.stem
    ).strip("_")[:60]

    interval = max(1, round(fps * INTERVAL_SECONDS))
    frame_number = 0
    saved = 0

    print("\nProcessing:", video_path.name)

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_number % interval == 0:
            timestamp = frame_number / fps

            filename = (
                f"video{video_index}_{safe_name}_"
                f"{timestamp:08.2f}s.jpg"
            )

            output_path = OUTPUT_DIR / filename

            if cv2.imwrite(str(output_path), frame):
                saved += 1

        frame_number += 1

    capture.release()
    total_saved += saved
    print("Frames saved:", saved)

print("\nEXTRACTION COMPLETED")
print("Total frames:", total_saved)
print("Output:", OUTPUT_DIR)
