"""
rebuild_dataset.py

Combines original WLASL frames (data/frames.npy) with newly downloaded YouTube
frames (data/youtube_frames.npy) into a single balanced dataset, then creates
fresh stratified train/val/test splits (70/15/15) and re-caches ResNet-50
features and per-frame MediaPipe landmarks.

Run AFTER collect_youtube_data.py.

Outputs (overwrite):
    data/frames.npy             : (N, 30, 224, 224, 3) all frames
    data/labels.npy             : (N,) int64 labels
    data/subsets.npy            : (N,) object 'train'/'val'/'test'
    data/resnet50_features.npy  : (N, 30, 2048) float32
    data/per_frame_landmarks.npy: (N, 30, 63) float32
    data/class_stats.json       : updated counts
    data/label_names.txt        : unchanged
"""

import os, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

WORDS       = ["drink", "computer", "before", "go", "who",
               "cousin", "help", "thin", "cool", "tall"]
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15   # remainder goes to test
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

# ── 1. Load and combine frames ───────────────────────────────
print("\n── 1. Loading frames ──")
wlasl_frames  = np.load("data/frames.npy")           # original WLASL frames
wlasl_labels  = np.load("data/labels.npy")

# Check if we have the original (pre-YouTube) WLASL data
# The original WLASL subsets.npy has the subset assignments
wlasl_subsets = np.load("data/subsets.npy", allow_pickle=True)

print(f"  WLASL frames:   {wlasl_frames.shape}  labels:{wlasl_labels.shape}")

if os.path.exists("data/youtube_frames.npy"):
    yt_frames = np.load("data/youtube_frames.npy")
    yt_labels = np.load("data/youtube_labels.npy")
    print(f"  YouTube frames: {yt_frames.shape}  labels:{yt_labels.shape}")
    all_frames = np.concatenate([wlasl_frames, yt_frames], axis=0)
    all_labels = np.concatenate([wlasl_labels, yt_labels], axis=0)
    # Mark source
    wlasl_source = np.full(len(wlasl_labels), "wlasl", dtype=object)
    yt_source    = np.full(len(yt_labels),    "youtube", dtype=object)
    source_all   = np.concatenate([wlasl_source, yt_source])
else:
    print("  No YouTube frames found — using WLASL only")
    all_frames = wlasl_frames
    all_labels = wlasl_labels
    source_all = np.full(len(wlasl_labels), "wlasl", dtype=object)

print(f"  Combined: {all_frames.shape}  N={len(all_labels)}")

# Count per class
print("  Per-class counts:")
for i, w in enumerate(WORDS):
    n = int((all_labels == i).sum())
    print(f"    {w:12s}: {n}")

# ── 2. Create fresh stratified split ────────────────────────
print("\n── 2. Creating stratified splits ──")
new_subsets = np.full(len(all_labels), "train", dtype=object)

for cls_idx in range(len(WORDS)):
    cls_mask    = (all_labels == cls_idx)
    cls_indices = np.where(cls_mask)[0]
    n           = len(cls_indices)

    # Shuffle reproducibly (seed + class index)
    rng = np.random.default_rng(SEED + cls_idx)
    perm = rng.permutation(n)
    cls_indices_shuffled = cls_indices[perm]

    n_train = max(1, int(round(n * TRAIN_FRAC)))
    n_val   = max(1, int(round(n * VAL_FRAC)))
    n_test  = n - n_train - n_val
    if n_test < 1:
        # Adjust: steal 1 from val if necessary
        n_val  = max(0, n_val - 1)
        n_test = n - n_train - n_val

    new_subsets[cls_indices_shuffled[:n_train]]                        = "train"
    new_subsets[cls_indices_shuffled[n_train:n_train + n_val]]         = "val"
    new_subsets[cls_indices_shuffled[n_train + n_val:]]                = "test"

    print(f"    {WORDS[cls_idx]:12s}: total={n}  train={n_train}  val={n_val}  test={n_test}")

tr_mask   = new_subsets == "train"
val_mask  = new_subsets == "val"
test_mask = new_subsets == "test"
print(f"  Totals → train:{tr_mask.sum()}  val:{val_mask.sum()}  test:{test_mask.sum()}")

# ── 3. Save updated frames/labels/subsets ───────────────────
print("\n── 3. Saving frames, labels, subsets ──")
np.save("data/frames.npy",  all_frames)
np.save("data/labels.npy",  all_labels)
np.save("data/subsets.npy", new_subsets)

stats = {w: {"train": 0, "val": 0, "test": 0} for w in WORDS}
for i, w in enumerate(WORDS):
    mask = all_labels == i
    stats[w]["train"] = int((mask & tr_mask).sum())
    stats[w]["val"]   = int((mask & val_mask).sum())
    stats[w]["test"]  = int((mask & test_mask).sum())
with open("data/class_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("  Saved frames.npy, labels.npy, subsets.npy, class_stats.json")

# ── 4. Re-cache ResNet-50 features ──────────────────────────
print("\n── 4. Re-caching ResNet-50 features ──")

class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames  # (N, 30, H, W, 3) float32 normalized
    def __len__(self): return len(self.frames)
    def __getitem__(self, idx):
        # (30, H, W, 3) -> (30, 3, H, W)
        return torch.from_numpy(self.frames[idx]).permute(0, 3, 1, 2)

rn50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
# Remove final FC layer to get 2048-d pool features
rn50_feat = nn.Sequential(*list(rn50.children())[:-1])  # up to avgpool

N = len(all_frames)
T = 30
features_out = np.zeros((N, T, 2048), dtype=np.float32)

BATCH_SIZE = 4  # videos per batch
loader = DataLoader(FrameDataset(all_frames), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=0)

idx_start = 0
t0 = time.time()
with torch.no_grad():
    for batch_idx, vid_batch in enumerate(loader):
        B = vid_batch.shape[0]
        # vid_batch: (B, T, 3, 224, 224)
        frames_flat = vid_batch.reshape(B * T, 3, 224, 224).to(device)
        feat_flat   = rn50_feat(frames_flat).squeeze(-1).squeeze(-1)  # (B*T, 2048)
        feat_vid    = feat_flat.cpu().numpy().reshape(B, T, 2048)
        features_out[idx_start:idx_start + B] = feat_vid
        idx_start  += B
        if (batch_idx + 1) % 20 == 0 or idx_start >= N:
            elapsed = time.time() - t0
            print(f"  ResNet features: {idx_start}/{N}  elapsed={elapsed:.0f}s")

np.save("data/resnet50_features.npy", features_out)
print(f"  Saved data/resnet50_features.npy  shape={features_out.shape}")

# ── 5. Re-extract per-frame MediaPipe landmarks ──────────────
print("\n── 5. Re-extracting per-frame landmarks ──")

MODEL_PATH = "./hand_landmarker.task"
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
hand_opts = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)

landmarks_out = np.zeros((N, T, 63), dtype=np.float32)
t0 = time.time()

with mp_vision.HandLandmarker.create_from_options(hand_opts) as detector:
    for i in range(N):
        for t in range(T):
            f    = all_frames[i, t]
            img  = np.clip(f * STD + MEAN, 0, 1)
            img  = (img * 255).astype(np.uint8)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            res    = detector.detect(mp_img)
            if res.hand_landmarks:
                lm  = res.hand_landmarks[0]
                vec = np.array([[p.x, p.y, p.z] for p in lm],
                               dtype=np.float32).flatten()
                landmarks_out[i, t] = vec
        if (i + 1) % 50 == 0 or i + 1 == N:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            print(f"  Landmarks: {i+1}/{N}  elapsed={elapsed:.0f}s  rate={rate:.1f} vid/s")

np.save("data/per_frame_landmarks.npy", landmarks_out)
nz = (landmarks_out.sum(axis=-1) != 0).sum()
print(f"  Saved data/per_frame_landmarks.npy  "
      f"hand frames={nz}/{N*T} ({nz/(N*T)*100:.1f}%)")

# ── 6. Also delete stale cached layer3 features ─────────────
if os.path.exists("data/layer3_features.npy"):
    os.remove("data/layer3_features.npy")
    print("\n  Deleted stale data/layer3_features.npy (will be regenerated if needed)")

print("\n── Done! ──")
print(f"  N={N}  train={tr_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}")
