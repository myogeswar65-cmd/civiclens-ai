
from pathlib import Path
from collections import defaultdict
from PIL import Image

root = Path("datasets/garbage_overflowing")
groups = defaultdict(list)
extensions = {".jpg", ".jpeg", ".png", ".webp"}

for split in ["train", "valid", "test"]:
    folder = root / split / "images"

    for f in folder.iterdir():
        if f.suffix.lower() not in extensions:
            continue

        # Group images that share the original filename before Roboflow's hash
        base = f.name.split(".rf.")[0]
        groups[base].append((split, f))

print("REPEATED SOURCE-NAME GROUPS")
found = 0

for base, items in groups.items():
    if len(items) > 1:
        found += 1
        print(base)
        for split, f in items:
            print(f"  {split}: {f.name}")

print("\nRepeated groups:", found)

print("\nIMAGE COLOR CHECK")
for split in ["train", "valid", "test"]:
    folder = root / split / "images"
    files = [f for f in folder.iterdir()
             if f.suffix.lower() in extensions]

    if files:
        with Image.open(files[0]) as img:
            print(f"{split}: mode={img.mode}, file={files[0].name}")
