
from pathlib import Path

root = Path("datasets/garbage_overflowing")
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

print("DATASET CHECK")
print("-" * 45)

for split in ["train", "valid", "test"]:
    folder = root / split
    images_dir = folder / "images"
    labels_dir = folder / "labels"

    images = [
        f for f in images_dir.glob("*")
        if f.suffix.lower() in image_extensions
    ]
    labels = list(labels_dir.glob("*.txt"))

    missing = [
        img.name for img in images
        if not (labels_dir / (img.stem + ".txt")).exists()
    ]

    print(f"{split}:")
    print(f"  Images: {len(images)}")
    print(f"  Labels: {len(labels)}")
    print(f"  Images without labels: {len(missing)}")

    if missing:
        print("  Examples:", missing[:5])
