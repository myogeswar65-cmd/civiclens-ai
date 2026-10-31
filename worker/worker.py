
from pathlib import Path
from collections import Counter
import base64
import json
import os
import time
import mimetypes
import subprocess

import cv2
import imageio_ffmpeg
from ultralytics import YOLO
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

ORIGINAL_MODEL = PROJECT_DIR / "models" / "best.pt"
CANDIDATE_MODEL = PROJECT_DIR / "models" / "candidate_best.pt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

MODEL_OPTIONS = {
    "original": ORIGINAL_MODEL,
    "candidate": CANDIDATE_MODEL,
}

ROBOFLOW_WORKSPACE = "m-yogeswar"
ROBOFLOW_WORKFLOW_ID = (
    "wastelens-bin-condition-zpqwq-1-rfdetr-small-t1-"
    "vwastelens-bin-condition-zpqwq-1-rfdetr-small-t1-logic"
)
ROBOFLOW_API_URL = "https://serverless.roboflow.com"


def load_model(model_choice="original"):
    if model_choice not in MODEL_OPTIONS:
        raise ValueError("Invalid local model choice.")

    model_path = MODEL_OPTIONS[model_choice]

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading {model_choice} model: {model_path}")

    model = YOLO(str(model_path))

    print("Model loaded successfully.")
    print("Classes:", model.names)

    return model, model_path


def load_roboflow_client():
    load_dotenv(PROJECT_DIR / ".env")

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY missing. Check the project .env file."
        )

    try:
        from inference_sdk import (
            InferenceHTTPClient,
            InferenceConfiguration,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install inference-sdk in your virtual environment."
        ) from exc

    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=api_key,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )

    return client


def roboflow_predict(client, frame, confidence):
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise RuntimeError("Could not encode video frame.")

    image_data = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    image_data_url = f"data:image/jpeg;base64,{image_data}"

    result = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={"image": image_data_url},
        use_cache=True,
    )

    if isinstance(result, list):
        output = result[0] if result else {}
    else:
        output = result

    prediction_block = output.get("predictions", {})

    predictions = (
        prediction_block.get("predictions", [])
        if isinstance(prediction_block, dict)
        else []
    )

    detections = []

    for prediction in predictions:
        score = float(prediction.get("confidence", 0))

        if score < confidence:
            continue

        center_x = float(prediction["x"])
        center_y = float(prediction["y"])
        box_width = float(prediction["width"])
        box_height = float(prediction["height"])

        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2

        class_name = prediction.get(
            "class",
            prediction.get("class_name", "unknown"),
        )

        detections.append({
            "class_id": prediction.get("class_id", -1),
            "class_name": str(class_name),
            "confidence": score,
            "box": [
                max(0, x1),
                max(0, y1),
                min(frame.shape[1], x2),
                min(frame.shape[0], y2),
            ],
        })

    return detections


def draw_detections(frame, detections):
    annotated = frame.copy()

    # OpenCV uses BGR color format.
    CLASS_COLORS = {
        "normal_bin": (0, 200, 0),
        "overflowing_bin": (0, 0, 255),
        "trash_outside_bin": (0, 165, 255),
    }

    for detection in detections:
        x1, y1, x2, y2 = [
            int(round(value)) for value in detection["box"]
        ]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        color = CLASS_COLORS.get(
            class_name,
            (255, 255, 255),
        )

        label = (
            f"{class_name.replace('_', ' ')} "
            f"{confidence:.0%}"
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )

        text_y = max(text_height + 10, y1)

        cv2.rectangle(
            annotated,
            (x1, text_y - text_height - 10),
            (x1 + text_width + 8, text_y + baseline - 5),
            color,
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 4, text_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def convert_to_browser_mp4(input_path, output_path):
    """
    Convert OpenCV's temporary MP4 into H.264 MP4
    for better compatibility with Chrome and HTML video players.
    """

    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise RuntimeError(
            "Temporary annotated video is missing or empty."
        )

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    print("Converting annotated video to H.264...")
    print(f"FFmpeg executable: {ffmpeg_exe}")

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    except subprocess.CalledProcessError as exc:
        error_details = exc.stderr or "Unknown FFmpeg error."

        raise RuntimeError(
            "H.264 video conversion failed:\n"
            + error_details[-3000:]
        ) from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            "H.264 conversion finished without a valid output file."
        )

    # Remove the temporary mp4v file after successful conversion.
    input_path.unlink(missing_ok=True)

    print(f"H.264 conversion successful: {output_path}")


def analyze_video(
    video_path: str,
    sample_every_n_frames: int = 5,
    model_choice: str = "original",
    confidence: float = 0.25,
) -> dict:

    path = Path(video_path)

    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported video format.")

    if sample_every_n_frames < 1:
        raise ValueError("Sampling interval must be at least 1.")

    if not 0.01 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0.01 and 1.0.")

    if model_choice not in ("original", "candidate", "roboflow"):
        raise ValueError("Invalid model choice.")

    model = None
    model_path = None
    roboflow_client = None

    if model_choice == "roboflow":
        roboflow_client = load_roboflow_client()

        model_description = (
            "Roboflow RF-DETR Small Serverless Cloud Workflow"
        )

        class_names = [
            "normal_bin",
            "overflowing_bin",
            "trash_outside_bin",
        ]

    else:
        model, model_path = load_model(model_choice)
        model_description = str(model_path)
        class_names = model.names

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError("Could not open the video file.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Could not read video dimensions.")

    tag = f"{path.stem}_{model_choice}"

    video_output = OUTPUT_DIR / f"{tag}_annotated.mp4"
    temp_video_output = OUTPUT_DIR / f"{tag}_temp.mp4"
    json_output = OUTPUT_DIR / f"{tag}_results.json"

    # OpenCV writes the temporary video using mp4v.
    writer = cv2.VideoWriter(
        str(temp_video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "Could not create temporary annotated video."
        )

    started = time.perf_counter()

    frame_number = 0
    sampled_frames = 0
    detections = []
    class_counts = Counter()

    # Hold the latest detections between cloud inference calls.
    latest_detections = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_number % sample_every_n_frames == 0:

                if model_choice == "roboflow":
                    current_detections = roboflow_predict(
                        roboflow_client,
                        frame,
                        confidence,
                    )

                else:
                    result = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=416,
                        verbose=False,
                    )[0]

                    current_detections = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            score = float(box.conf.item())
                            class_name = model.names[class_id]

                            coordinates = [
                                round(float(value), 2)
                                for value in box.xyxy[0].tolist()
                            ]

                            current_detections.append({
                                "class_id": class_id,
                                "class_name": str(class_name),
                                "confidence": score,
                                "box": coordinates,
                            })

                latest_detections = current_detections
                sampled_frames += 1

                for detection in current_detections:
                    detections.append({
                        "frame": frame_number,
                        **detection,
                    })

                    class_counts[detection["class_name"]] += 1

            annotated_frame = draw_detections(
                frame,
                latest_detections,
            )

            writer.write(annotated_frame)

            frame_number += 1

    finally:
        capture.release()
        writer.release()

    # Convert the completed temporary video into H.264.
    convert_to_browser_mp4(
        temp_video_output,
        video_output,
    )

    elapsed = time.perf_counter() - started

    report = {
        "filename": path.name,
        "status": "completed",
        "model_choice": model_choice,
        "model": model_description,
        "classes": class_names,
        "video_fps": round(fps, 2),
        "total_video_frames": frame_count,
        "processed_frames": sampled_frames,
        "sampling_interval": sample_every_n_frames,
        "confidence_threshold": confidence,
        "detection_count": len(detections),
        "detections_by_class": dict(class_counts),
        "processing_time_seconds": round(elapsed, 2),
        "inference_device": (
            "Roboflow cloud"
            if model_choice == "roboflow"
            else "Ultralytics automatic device selection"
        ),
        "annotated_video": video_output.name,
        "results_file": json_output.name,
        "detections": detections,
    }

    json_output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Analysis completed. Frames sampled: {sampled_frames}")
    print(f"Detections: {len(detections)}")
    print(f"Annotated video: {video_output}")
    print(f"JSON report: {json_output}")

    return report


if __name__ == "__main__":
    print("WasteLens Predict - video inference worker")

    for name, path in MODEL_OPTIONS.items():
        print(
            f"{name}: "
            f"{'available' if path.is_file() else 'missing'}"
        )

    print("roboflow: cloud workflow configured")
