
from pathlib import Path

root = Path("datasets/garbage_overflowing")
names = ["trash onbin", "trash out bin", "trash-bin"]
counts = [0, 0, 0]
errors = []

for split in ["train", "valid", "test"]:
    labels_dir = root / split / "labels"

    for file in labels_dir.glob("*.txt"):
        for line_no, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), 1
        ):
            parts = line.split()

            try:
                assert len(parts) == 5, "Expected 5 values"
                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]

                assert 0 <= class_id < 3, "Invalid class ID"
                assert all(0 <= v <= 1 for v in coords), "Coordinate outside 0-1"
                assert coords[2] > 0 and coords[3] > 0, "Invalid box size"

                counts[class_id] += 1

            except (ValueError, AssertionError) as exc:
                errors.append(
                    f"{split}/{file.name}:{line_no} - {exc}: {line}"
                )

print("ANNOTATION VALIDATION")
print("-" * 40)
print("Bounding boxes by class:")
for name, count in zip(names, counts):
    print(f"  {name}: {count}")

print(f"\nInvalid annotations: {len(errors)}")
for error in errors[:10]:
    print(error)
