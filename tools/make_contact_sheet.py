
from pathlib import Path
from PIL import Image, ImageDraw
import math

BASE = Path(__file__).resolve().parent
FRAME_DIR = BASE / "datasets/bin_condition/images/raw_frames"
OUTPUT = BASE / "datasets/bin_condition/review"

OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(FRAME_DIR.glob("*.jpg"))

if not files:
    raise FileNotFoundError("No extracted frames found.")

columns = 4
thumb_width = 320
thumb_height = 200
label_height = 35
rows = math.ceil(len(files) / columns)

sheet = Image.new(
    "RGB",
    (columns * thumb_width, rows * (thumb_height + label_height)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, path in enumerate(files):
    image = Image.open(path).convert("RGB")
    image.thumbnail((thumb_width, thumb_height))

    x = (index % columns) * thumb_width
    y = (index // columns) * (thumb_height + label_height)

    sheet.paste(image, (x, y))
    draw.text(
        (x + 5, y + thumb_height + 5),
        f"{index + 1}. {path.name[:38]}",
        fill="black"
    )

sheet.save(OUTPUT / "all_frames_overview.jpg", quality=90)

print("Contact sheet created successfully.")
print("Frames:", len(files))
print("Saved:", OUTPUT / "all_frames_overview.jpg")
