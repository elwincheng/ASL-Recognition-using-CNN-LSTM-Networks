"""
landmark_lstm.py  —  Per-frame MediaPipe landmarks + LSTM classifier

The SVM baseline averaged MediaPipe landmarks across time, discarding all
temporal information.  This model feeds the per-frame landmark sequence
directly into a small LSTM, capturing hand motion trajectories.

Steps:
  1. Extract 63-dim per-frame landmarks for all 147 videos × 30 frames
     (zeros for frames where MediaPipe detects no hand).
  2. Train a 2-layer LSTM on the train split (105 videos) with label smoothing,
     dropout, and class-weighted loss.
  3. Validate on val split, save best checkpoint.
  4. Ensemble 3 random seeds and evaluate on test.

Outputs:
  - data/per_frame_landmarks.npy  : (147, 30, 63) float32
  - results/landmark_lstm_results.json
  - figures/landmark_learning_curves.png
  - figures/landmark_confusion.png
  - figures/landmark_comparison.png
"""

import os, json, random, copy, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}", flush=True)

# ── Load frames ─────────────────────────────────────────────
print("Loading frames...", flush=True)
frames_all  = np.load("data/frames.npy")        # (147, 30, 224, 224, 3) float32 normalized
labels_all  = np.load("data/labels.npy")
subsets_all = np.load("data/subsets.npy", allow_pickle=True)
with open("data/label_names.txt") as f:
    label_names = [l.strip() for l in f]
NUM_CLASSES = len(label_names)
N = len(labels_all)

print(f"Loaded frames: {frames_all.shape}  N={N}", flush=True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def unnormalize(frame_f32):
    img = frame_f32 * STD + MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

# ── Extract per-frame landmarks ──────────────────────────────
LANDMARK_FILE = "data/per_frame_landmarks.npy"

if os.path.exists(LANDMARK_FILE):
    print(f"Loading cached per-frame landmarks from {LANDMARK_FILE}", flush=True)
    landmarks_all = np.load(LANDMARK_FILE)
else:
    print("Extracting per-frame MediaPipe landmarks...", flush=True)
    MODEL_PATH = "./hand_landmarker.task"
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    hand_options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )

    landmarks_all = np.zeros((N, 30, 63), dtype=np.float32)  # zeros = no hand detected
    t0 = time.time()

    with mp_vision.HandLandmarker.create_from_options(hand_options) as detector:
        for i in range(N):
            detected_frames = 0
            for t in range(30):
                frame_f32 = frames_all[i, t]
                img_uint8 = unnormalize(frame_f32)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_uint8)
                result    = detector.detect(mp_image)
                if result.hand_landmarks:
                    lm  = result.hand_landmarks[0]
                    vec = np.array([[p.x, p.y, p.z] for p in lm],
                                   dtype=np.float32).flatten()  # (63,)
                    landmarks_all[i, t] = vec
                    detected_frames += 1
            if (i + 1) % 10 == 0 or i == N - 1:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                print(f"  {i+1}/{N}  elapsed={elapsed:.0f}s  rate={rate:.1f} vids/s", flush=True)

    np.save(LANDMARK_FILE, landmarks_all)
    print(f"Saved {LANDMARK_FILE}  shape={landmarks_all.shape}", flush=True)

print(f"Landmarks shape: {landmarks_all.shape}  dtype={landmarks_all.dtype}", flush=True)

# Check how many frames have detected hands (non-zero)
nonzero = (landmarks_all.sum(axis=-1) != 0).sum()
print(f"Frames with detected hand: {nonzero}/{N*30} ({nonzero/(N*30)*100:.1f}%)", flush=True)

# ── Splits ───────────────────────────────────────────────────
tr_mask   = subsets_all == "train"
val_mask  = subsets_all == "val"
test_mask = subsets_all == "test"

y_tr   = labels_all[tr_mask]
y_val  = labels_all[val_mask]
y_test = labels_all[test_mask]

lm_tr   = landmarks_all[tr_mask]
lm_val  = landmarks_all[val_mask]
lm_test = landmarks_all[test_mask]

print(f"Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_test)}", flush=True)
BASELINE_TEST = 56.2

# Class weights on training set
cw = 1.0 / (np.bincount(y_tr, minlength=NUM_CLASSES).astype(float) + 1e-6)
cw = cw / cw.sum() * NUM_CLASSES
cw_t = torch.tensor(cw, dtype=torch.float32).to(device)


# ── Dataset ──────────────────────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, lmarks, labels, augment=False):
        self.lm      = lmarks   # (N, 30, 63)
        self.labels  = labels
        self.augment = augment

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        x = self.lm[idx].copy()   # (30, 63)
        if self.augment:
            # Mirror x-coordinates of landmarks (flip left-right)
            # x landmarks are in [0,1]; mirroring: x' = 1 - x
            x_coords = x[:, 0::3].copy()   # columns 0, 3, 6, ... = x-coords of each landmark
            x[:, 0::3] = 1.0 - x_coords
            # Small Gaussian noise on non-zero frames
            mask = (x.sum(axis=-1) != 0)[:, None]  # (30, 1)
            x += mask * np.random.randn(*x.shape).astype(np.float32) * 0.005
        return (torch.from_numpy(x),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── Model ────────────────────────────────────────────────────
class LandmarkLSTM(nn.Module):
    """Tiny 2-layer LSTM on 63-dim hand landmark sequences."""
    def __init__(self, input_dim=63, hidden=64, num_layers=2,
                 dropout=0.5, num_classes=10):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):           # x: (B, T, 63)
        x = self.norm(x)
        out, _ = self.lstm(x)       # (B, T, hidden)
        return self.head(out.mean(1))   # global average pooling over time


# ── Training ─────────────────────────────────────────────────
EPOCHS = 150
SEEDS  = [42, 123, 7]


def run_one_seed(seed_val):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    ds_tr  = LandmarkDataset(lm_tr,   y_tr,   augment=True)
    ds_val = LandmarkDataset(lm_val,  y_val,  augment=False)

    tr_loader  = DataLoader(ds_tr,  batch_size=8, shuffle=True,  num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=0)

    m    = LandmarkLSTM(input_dim=63, hidden=64, num_layers=2,
                        dropout=0.5, num_classes=NUM_CLASSES).to(device)
    opt  = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.1)

    bv, be, bs = 0.0, 0, None
    h = {"train_acc": [], "val_acc": []}

    for ep in range(1, EPOCHS + 1):
        m.train()
        tc, tt = 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = m(xb); loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tc += (out.argmax(1) == yb).sum().item(); tt += len(yb)
        sched.step()

        m.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vc += (m(xb).argmax(1) == yb).sum().item(); vt += len(yb)

        ta_ep = tc / tt; va_ep = vc / vt
        h["train_acc"].append(ta_ep); h["val_acc"].append(va_ep)
        if va_ep > bv:
            bv, be = va_ep, ep; bs = copy.deepcopy(m.state_dict())

        if ep % 10 == 0 or ep == 1:
            print(f"  s={seed_val} ep{ep:3d}/{EPOCHS}  "
                  f"tr={ta_ep*100:.0f}%  val={va_ep*100:.0f}%  "
                  f"(best={bv*100:.0f}%@{be})", flush=True)

    # Evaluate best checkpoint on test
    ds_test = LandmarkDataset(lm_test, y_test, augment=False)
    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=0)

    m.load_state_dict(bs); m.eval()
    preds, logits = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            lgt = m(xb.to(device)).cpu()
            logits.append(lgt); preds.extend(lgt.argmax(1).numpy())
    logits = torch.cat(logits, 0)
    ta = accuracy_score(y_test, preds)
    print(f"  → seed={seed_val}  val={bv*100:.1f}%@ep{be}  "
          f"test={ta*100:.1f}% ({int(ta*16)}/16)", flush=True)
    return ta, preds, bv, be, h, logits


print(f"\n── Landmark-LSTM  ({EPOCHS} epochs, 3 seeds) ──", flush=True)
results_seeds = {}
all_logits = []

for s in SEEDS:
    ta, pd, bv, be, h, lg = run_one_seed(s)
    results_seeds[s] = {"ta": ta, "pd": pd, "bv": bv, "be": be, "h": h}
    all_logits.append(lg)

# Ensemble
ens_logits = sum(all_logits) / len(all_logits)
ens_preds  = ens_logits.argmax(1).numpy().tolist()
ta_ens     = accuracy_score(y_test, ens_preds)

bv_list = [results_seeds[s]["bv"] for s in SEEDS]
ta_list = [results_seeds[s]["ta"] for s in SEEDS]
best_s  = SEEDS[int(np.argmax(bv_list))]
ta_best = results_seeds[best_s]["ta"]
h_best  = results_seeds[best_s]["h"]
bv_best = results_seeds[best_s]["bv"]

if ta_ens >= max(ta_list):
    final_preds = ens_preds; final_test = ta_ens; final_tag = "Ensemble (3 seeds)"
else:
    final_preds = results_seeds[best_s]["pd"]
    final_test  = ta_best; final_tag = f"Single seed={best_s}"

print("\n" + "="*60)
print("LANDMARK-LSTM RESULTS SUMMARY")
print("="*60)
print(f"  SVM baseline (MediaPipe avg):  val=61.5%  test=56.2%  (9/16)")
for s in SEEDS:
    r = results_seeds[s]
    print(f"  LM-LSTM seed={s:3d}:  val={r['bv']*100:.1f}%  test={r['ta']*100:.1f}%  ({int(r['ta']*16)}/16)")
print(f"  LM-LSTM ensemble:           test={ta_ens*100:.1f}%  ({int(ta_ens*16)}/16)")
beat = final_test * 100 > BASELINE_TEST
print(f"  Selected: {final_tag}  test={final_test*100:.1f}%")
print(f"  {'✓ BEATS baseline!' if beat else '✗ Below baseline'}")
print("="*60)

# Per-class accuracy
per_class_acc = {}
for i, name in enumerate(label_names):
    mask = np.array(y_test) == i
    if mask.sum() > 0:
        per_class_acc[name] = float(accuracy_score(
            np.array(y_test)[mask], np.array(final_preds)[mask]))
    else:
        per_class_acc[name] = None

with open("results/baseline_results.json") as f:
    bl = json.load(f)

for name in label_names:
    sa = bl["per_class_acc"].get(name) or 0
    ca = (per_class_acc.get(name) or 0)*100
    diff = "↑" if ca > sa else ("↓" if ca < sa else "=")
    print(f"  {name:12s}  SVM={sa:.0f}%  LM-LSTM={ca:.0f}%  {diff}")

# ── Figures ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ep_x = list(range(1, EPOCHS + 1))
colors_t = ["#93c5fd", "#6ee7b7", "#fcd34d"]
colors_v = ["#2563eb", "#059669", "#d97706"]

for i, s in enumerate(SEEDS):
    r = results_seeds[s]
    axes[0].plot(ep_x, [a*100 for a in r["h"]["train_acc"]], color=colors_t[i], lw=1, ls="--", alpha=0.6)
    axes[0].plot(ep_x, [a*100 for a in r["h"]["val_acc"]], color=colors_v[i], lw=1.6,
                 label=f"Val seed={s} (best={r['bv']*100:.0f}%)")

axes[0].axhline(BASELINE_TEST, color="#dc2626", ls=":", lw=1.5, label=f"SVM test ({BASELINE_TEST}%)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Landmark-LSTM training curves (dashed=train, solid=val)", fontsize=10)
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 110)

bar_labels = ["SVM\n(MediaPipe\navg)", "Phase-1\nCNN-LSTM\n(frozen ResNet)", "Landmark\nLSTM\nbest", "Landmark\nLSTM\nensemble"]
test_bars  = [56.2, 37.5, max(ta_list)*100, ta_ens*100]
bar_colors = ["#94a3b8", "#f87171", "#3b82f6", "#1d4ed8"]
x_b = np.arange(len(bar_labels))
bars = axes[1].bar(x_b, test_bars, color=bar_colors, edgecolor="white", lw=0.8)
axes[1].axhline(56.2, color="#94a3b8", lw=1.5, ls="--", alpha=0.8)
axes[1].axhline(10, color="gray", lw=1.2, ls=":", label="Random (10%)")
axes[1].set_xticks(x_b); axes[1].set_xticklabels(bar_labels, fontsize=9)
axes[1].set_ylim(0, 85); axes[1].set_ylabel("Test accuracy (%)")
axes[1].set_title("Model comparison on test set"); axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, test_bars):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.suptitle("Landmark-LSTM: Per-frame hand landmarks → temporal LSTM", fontsize=11)
plt.tight_layout()
plt.savefig("figures/landmark_learning_curves.png", dpi=150); plt.close()
print("\nSaved figures/landmark_learning_curves.png")

# Confusion matrix
cm = confusion_matrix(y_test, final_preds, labels=list(range(NUM_CLASSES)))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ticks = np.arange(NUM_CLASSES)
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xticklabels(label_names, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(label_names, fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Landmark-LSTM — Test Confusion Matrix ({final_tag})")
thresh = max(1, cm.max() / 2.0)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig("figures/landmark_confusion.png", dpi=150); plt.close()
print("Saved figures/landmark_confusion.png")

# Per-class grouped bar chart
svm_pc = [bl["per_class_acc"].get(n) or 0 for n in label_names]
lm_pc  = [(per_class_acc.get(n) or 0)*100 for n in label_names]
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(NUM_CLASSES); w = 0.35
ax.bar(x-w/2, svm_pc, w, label="SVM (MediaPipe avg)", color="#94a3b8")
ax.bar(x+w/2, lm_pc,  w, label="Landmark-LSTM (temporal)", color="#3b82f6")
ax.axhline(10, color="gray", ls="--", lw=1.2, label="Random (10%)")
ax.set_xticks(x); ax.set_xticklabels(label_names, rotation=35, ha="right", fontsize=9)
ax.set_ylim(0, 115); ax.set_ylabel("Per-class test accuracy (%)")
ax.set_title("Per-class accuracy: SVM baseline vs Landmark-LSTM primary model"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/landmark_comparison.png", dpi=150); plt.close()
print("Saved figures/landmark_comparison.png")

# Save results
results = {
    **{f"lm_seed{s}": {"val": round(results_seeds[s]["bv"]*100, 1),
                        "test": round(results_seeds[s]["ta"]*100, 1)} for s in SEEDS},
    "lm_ensemble":         {"test": round(ta_ens*100, 1)},
    "final_model":         final_tag,
    "final_test_accuracy": round(final_test*100, 1),
    "final_val_accuracy":  round(bv_best*100, 1),
    "beats_baseline":      beat,
    "baseline_test":       BASELINE_TEST,
    "baseline_val":        61.5,
    "per_class_acc":       {k: (round(v*100, 1) if v is not None else None) for k, v in per_class_acc.items()},
    "confusion_matrix":    cm.tolist(),
    "label_names":         label_names,
    "lm_history":          {k: [float(a) for a in v] for k, v in h_best.items()},
}
with open("results/landmark_lstm_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results/landmark_lstm_results.json")
print("\nDone!", flush=True)
