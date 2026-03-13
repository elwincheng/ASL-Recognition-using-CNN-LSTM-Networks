"""
data_processing.py

Loads the WLASL-processed dataset (from Kaggle cache), selects 10 classes
with the most available videos, extracts 30 uniformly-sampled frames per video,
and saves everything as a numpy array for downstream use.

Output files (saved to ./data/):
  - frames.npy        : float32 array (N, 30, 224, 224, 3), normalized to [0,1]
  - labels.npy        : int64 array (N,)
  - subsets.npy       : str array (N,) - 'train', 'val', or 'test'
  - label_names.txt   : one class name per line
  - class_stats.json  : per-class counts

Usage:
  python data_processing.py
"""

import os, json, cv2
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DATA_PATH   = "/Users/elwincheng/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5"
OUT_DIR     = "./data"
NUM_FRAMES  = 30
IMG_SIZE    = 224
NUM_CLASSES = 10
# ImageNet mean / std (per channel)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# Load metadata
# ─────────────────────────────────────────
with open(f"{DATA_PATH}/wlasl_class_list.txt") as f:
    class_list = [line.strip().split("\t")[1] for line in f]

with open(f"{DATA_PATH}/nslt_100.json") as f:
    nslt = json.load(f)

# ─────────────────────────────────────────
# Find top-N classes by number of available videos
# (require at least 1 test video so we can evaluate)
# ─────────────────────────────────────────
class_vids = defaultdict(list)
for vid_id, info in nslt.items():
    vid_path = f"{DATA_PATH}/videos/{vid_id}.mp4"
    if os.path.exists(vid_path):
        label_idx = info["action"][0]
        class_vids[label_idx].append((vid_id, info["subset"]))

# Filter: must have at least 1 test video
eligible = {
    idx: vids for idx, vids in class_vids.items()
    if any(s == "test" for _, s in vids)
}

# Pick top NUM_CLASSES by total video count
top_classes = sorted(eligible, key=lambda i: -len(eligible[i]))[:NUM_CLASSES]
top_classes = sorted(top_classes)  # sort by original label index for reproducibility

print(f"Selected {NUM_CLASSES} classes:")
for i, idx in enumerate(top_classes):
    vids = eligible[idx]
    n_tr = sum(1 for _, s in vids if s == "train")
    n_va = sum(1 for _, s in vids if s == "val")
    n_te = sum(1 for _, s in vids if s == "test")
    print(f"  [{i}] {class_list[idx]:20s} | total={len(vids):3d}  train={n_tr}  val={n_va}  test={n_te}")

label_names = [class_list[idx] for idx in top_classes]
label_map   = {orig_idx: new_idx for new_idx, orig_idx in enumerate(top_classes)}

# ─────────────────────────────────────────
# Helper: extract N uniformly-sampled frames
# ─────────────────────────────────────────
def extract_frames(video_path, n_frames=NUM_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    # Build a list of frame indices to sample
    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        # Loop the video if it's too short
        indices = [i % total for i in range(n_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Try reading sequentially as fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            # ImageNet normalisation (helps ResNet transfer learning)
            frame = (frame - MEAN) / STD
            frames.append(frame)
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))

    cap.release()
    return np.stack(frames)  # (n_frames, H, W, 3)


# ─────────────────────────────────────────
# Extract frames for all selected videos
# ─────────────────────────────────────────
all_frames  = []
all_labels  = []
all_subsets = []
stats       = {name: {"train": 0, "val": 0, "test": 0} for name in label_names}

total_vids = sum(len(eligible[idx]) for idx in top_classes)
processed  = 0

for orig_idx in top_classes:
    new_idx = label_map[orig_idx]
    name    = class_list[orig_idx]
    for vid_id, subset in eligible[orig_idx]:
        vid_path = f"{DATA_PATH}/videos/{vid_id}.mp4"
        frames   = extract_frames(vid_path)
        if frames is None:
            continue
        all_frames.append(frames)
        all_labels.append(new_idx)
        all_subsets.append(subset)
        stats[name][subset] += 1
        processed += 1
        print(f"  [{processed}/{total_vids}] {vid_id}.mp4  → class={name}  subset={subset}")

all_frames  = np.stack(all_frames).astype(np.float32)   # (N, 30, 224, 224, 3)
all_labels  = np.array(all_labels,  dtype=np.int64)
all_subsets = np.array(all_subsets, dtype=object)

print(f"\nTotal videos loaded: {len(all_labels)}")
print("Shape:", all_frames.shape)

# ─────────────────────────────────────────
# Save
# ─────────────────────────────────────────
np.save(f"{OUT_DIR}/frames.npy",  all_frames)
np.save(f"{OUT_DIR}/labels.npy",  all_labels)
np.save(f"{OUT_DIR}/subsets.npy", all_subsets)

with open(f"{OUT_DIR}/label_names.txt", "w") as f:
    for name in label_names:
        f.write(name + "\n")

with open(f"{OUT_DIR}/class_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\nClass statistics:")
for name, s in stats.items():
    print(f"  {name:20s}  train={s['train']}  val={s['val']}  test={s['test']}")
print("\nDone. Files saved to", OUT_DIR)
