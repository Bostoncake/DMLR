from datasets import load_dataset
from pathlib import Path
import json
import shutil

# 1) load dataset
ds = load_dataset("MMVP/MMVP")

# 2) choose split
split_name = list(ds.keys())[0]   # or set manually, e.g. "test"
split = ds[split_name]

# 3) inspect columns once if needed
print(split.features)

# change this if your image column is not named "image"
image_col = "image"

# 4) output folders
img_dir = Path("dataset/mmvp/MMVP Images")
img_dir.mkdir(parents=True, exist_ok=True)

records = []

for idx, ex in enumerate(split):
    img = ex[image_col]

    # save as 1.jpg, 2.jpg, ...
    img_filename = f"{idx + 1}.jpg"
    img_path = img_dir / img_filename

    # if img is a PIL image
    if hasattr(img, "save"):
        img.convert("RGB").save(img_path, format="JPEG")
    # if img is a dict with cached path
    elif isinstance(img, dict) and img.get("path") is not None:
        shutil.copy2(img["path"], img_path)
    else:
        raise ValueError(f"Unsupported image format at index {idx}: {type(img)}")

    records.append({
        "image_path": f"dataset/mmvp/MMVP Images/{img_filename}"
    })

# 5) save json
with open("dataset/mmvp/mmvp.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Done. Exported {len(records)} samples.")
print("Images saved to:", img_dir)
print("Metadata saved to: dataset/mmvp/mmvp.json")