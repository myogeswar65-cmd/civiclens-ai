
import os
import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)

# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY missing. Check your .env file."
    )

# --------------------------------------------------
# 2. Roboflow workflow configuration
# --------------------------------------------------

WORKSPACE = "m-yogeswar"

WORKFLOW_ID = (
    "wastelens-bin-condition-zpqwq-1-rfdetr-small-t1-"
    "vwastelens-bin-condition-zpqwq-1-rfdetr-small-t1-logic"
)

API_URL = "https://serverless.roboflow.com"

# --------------------------------------------------
# 3. Select test image
# --------------------------------------------------

IMAGE_DIR = Path(
    "datasets/bin_condition/images/raw_frames_v2"
)

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    raise FileNotFoundError(
        f"No JPG images found in: {IMAGE_DIR.resolve()}"
    )

# Select the 17th image if available
test_image = images[16] if len(images) > 16 else images[0]

if not test_image.exists():
    raise FileNotFoundError(test_image)

print("\nSelected image:", test_image)
print("Image exists:", test_image.exists())
print("File size:", test_image.stat().st_size, "bytes")

# --------------------------------------------------
# 4. Convert image to Base64 data URL
# --------------------------------------------------

mime_type = mimetypes.guess_type(test_image)[0]

if not mime_type:
    mime_type = "image/jpeg"

with open(test_image, "rb") as file:
    encoded_image = base64.b64encode(
        file.read()
    ).decode("utf-8")

image_data_url = (
    f"data:{mime_type};base64,{encoded_image}"
)

# --------------------------------------------------
# 5. Connect to Roboflow
# --------------------------------------------------

client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)

# --------------------------------------------------
# 6. Run workflow
# --------------------------------------------------

print("\nSending image to Roboflow...")

result = client.run_workflow(
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW_ID,
    images={
        "image": image_data_url
    },
    use_cache=True
)

# --------------------------------------------------
# 7. Display results
# --------------------------------------------------


# --------------------------------------------------
# 7. Draw predictions on the original image
# --------------------------------------------------

import cv2

predictions = result[0]["predictions"]["predictions"]

image = cv2.imread(str(test_image))

if image is None:
    raise ValueError("Could not open test image")

for pred in predictions:
    x = pred["x"]
    y = pred["y"]
    w = pred["width"]
    h = pred["height"]

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    class_name = pred["class"]
    confidence = pred["confidence"]

    label = f"{class_name} {confidence:.2%}"

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        image,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# Save annotated image
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

output_path = OUTPUT_DIR / "roboflow_test_annotated.jpg"

cv2.imwrite(str(output_path), image)

print("\nDetections:", len(predictions))

for pred in predictions:
    print(
        pred["class"],
        f'{pred["confidence"]:.2%}'
    )

print("\nAnnotated image saved:", output_path)
