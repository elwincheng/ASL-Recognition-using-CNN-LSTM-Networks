"""
baseline_model.py

Baseline: MediaPipe Hands landmark extraction + SVM classifier.

For each video we extract 21 hand landmarks (x, y, z) from every sampled frame
using MediaPipe HandLandmarker, then time-average them to get a single 63-dim
feature vector. An RBF-kernel SVM is tuned with cross-validation on the training
set and evaluated on the test set.

Outputs:
  - figures/baseline_confusion.png     : test-set confusion matrix
  - figures/sample_frames_landmarks.png: sample frames with drawn landmarks
  - figures/baseline_per_class.png     : per-class accuracy bar chart
  - results/baseline_results.json      : all accuracy numbers
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)

MODEL_PATH = "./hand_landmarker.task"

# ─────────────────────────────────────────────────────────
# Load pre-processed data
# ─────────────────────────────────────────────────────────
frames_all  = np.load("data/frames.npy")        # (N, 30, 224, 224, 3)
labels_all  = np.load("data/labels.npy")
subsets_all = np.load("data/subsets.npy", allow_pickle=True)

with open("data/label_names.txt") as f:
    label_names = [l.strip() for l in f]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def unnormalize(frame_f32):
    """Reverse ImageNet normalization → uint8 RGB."""
    img = frame_f32 * STD + MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────
# MediaPipe HandLandmarker (new Tasks API)
# ─────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
hand_options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)

def extract_landmarks_video(frames_seq):
    """
    frames_seq : np.ndarray (30, 224, 224, 3) normalized float32
    Returns    : (63,) averaged landmark vector (zeros if no hand found)
    """
    accumulated = []
    with mp_vision.HandLandmarker.create_from_options(hand_options) as detector:
        for frame_f32 in frames_seq:
            img_uint8 = unnormalize(frame_f32)
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=img_uint8
            )
            result = detector.detect(mp_image)
            if result.hand_landmarks:
                # Use first hand detected
                lm  = result.hand_landmarks[0]
                vec = np.array([[p.x, p.y, p.z] for p in lm],
                               dtype=np.float32).flatten()  # (63,)
                accumulated.append(vec)

    if len(accumulated) == 0:
        return np.zeros(63, dtype=np.float32), 0
    return np.mean(accumulated, axis=0), len(accumulated)


print("Extracting MediaPipe landmarks…")
n        = len(frames_all)
features = np.zeros((n, 63), dtype=np.float32)
detected = []        # frames with at least one hand detected

for i in range(n):
    feat, n_detected = extract_landmarks_video(frames_all[i])
    features[i]      = feat
    detected.append(n_detected)
    if (i + 1) % 20 == 0 or i == n - 1:
        print(f"  {i+1}/{n}  avg frames with hand detected: "
              f"{np.mean(detected[-20:]):.1f}/30")

no_hand = sum(1 for d in detected if d == 0)
print(f"\nVideos with NO hand detected at all: {no_hand}/{n}")

# ─────────────────────────────────────────────────────────
# Train / val / test split
# ─────────────────────────────────────────────────────────
tr_mask   = subsets_all == "train"
val_mask  = subsets_all == "val"
test_mask = subsets_all == "test"

X_train, y_train = features[tr_mask],   labels_all[tr_mask]
X_val,   y_val   = features[val_mask],  labels_all[val_mask]
X_test,  y_test  = features[test_mask], labels_all[test_mask]

print(f"\nSplit sizes → train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

# Scale features
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────
# SVM hyper-parameter search
# ─────────────────────────────────────────────────────────
print("\nRunning grid search (C × gamma, 5-fold CV on train set)…")
param_grid = {
    "C":     [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.001],
}
svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
gs  = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy",
                   n_jobs=-1, refit=True)
gs.fit(X_train_sc, y_train)
best_svm = gs.best_estimator_

print(f"Best params:  {gs.best_params_}")
print(f"CV accuracy:  {gs.best_score_*100:.1f}%")

# ─────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────
val_acc     = accuracy_score(y_val,  best_svm.predict(X_val_sc))
y_pred_test = best_svm.predict(X_test_sc)
test_acc    = accuracy_score(y_test, y_pred_test)

print(f"Val  accuracy: {val_acc*100:.1f}%")
print(f"Test accuracy: {test_acc*100:.1f}%")
print("\nPer-class test report:")
print(classification_report(y_test, y_pred_test,
                             target_names=label_names, zero_division=0))

per_class_acc = {}
for cls_idx, cls_name in enumerate(label_names):
    mask = y_test == cls_idx
    if mask.sum() > 0:
        per_class_acc[cls_name] = float(accuracy_score(y_test[mask], y_pred_test[mask]))
    else:
        per_class_acc[cls_name] = None

# ─────────────────────────────────────────────────────────
# Figure 1: Confusion matrix
# ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_test, labels=list(range(len(label_names))))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ticks = np.arange(len(label_names))
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xticklabels(label_names, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(label_names, fontsize=9)
ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("True", fontsize=10)
ax.set_title("SVM Baseline — Test Confusion Matrix", fontsize=11)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig("figures/baseline_confusion.png", dpi=150)
plt.close()
print("Saved figures/baseline_confusion.png")

# ─────────────────────────────────────────────────────────
# Figure 2: Sample frames with landmark overlay (3 classes × 5 frames)
# ─────────────────────────────────────────────────────────
classes_to_show = [0, 2, 5]  # drink, before, cousin
fig, axes = plt.subplots(3, 5, figsize=(10, 6.5))

# Mediapipe drawing utils (still available via tasks.python.vision)
connections = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS

def draw_landmarks_on_image(rgb_img, hand_landmarks):
    """Draw hand skeleton on an RGB uint8 image."""
    img_copy = rgb_img.copy()
    h, w = img_copy.shape[:2]
    # Draw connections
    for conn in connections:
        start = hand_landmarks[conn.start]
        end   = hand_landmarks[conn.end]
        x0, y0 = int(start.x * w), int(start.y * h)
        x1, y1 = int(end.x * w), int(end.y * h)
        cv2.line(img_copy, (x0, y0), (x1, y1), (0, 220, 0), 1)
    # Draw points
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img_copy, (cx, cy), 3, (255, 50, 50), -1)
    return img_copy

shown_frames = [0, 7, 14, 21, 28]

with mp_vision.HandLandmarker.create_from_options(hand_options) as detector:
    for row, cls_idx in enumerate(classes_to_show):
        mask  = (labels_all == cls_idx) & tr_mask
        idxs  = np.where(mask)[0]
        vid_i = idxs[0]
        for col, fnum in enumerate(shown_frames):
            frame_f32 = frames_all[vid_i, fnum]
            img_rgb   = unnormalize(frame_f32)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = detector.detect(mp_image)

            if result.hand_landmarks:
                img_show = draw_landmarks_on_image(img_rgb, result.hand_landmarks[0])
            else:
                img_show = img_rgb

            axes[row, col].imshow(img_show)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(
                    label_names[cls_idx], fontsize=10,
                    rotation=0, labelpad=45, va="center"
                )
            if row == 0:
                axes[row, col].set_title(f"frame {fnum+1}", fontsize=9)

plt.suptitle(
    "Sample cleaned frames with MediaPipe hand landmarks (train split)",
    fontsize=11, y=1.01
)
plt.tight_layout()
plt.savefig("figures/sample_frames_landmarks.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/sample_frames_landmarks.png")

# ─────────────────────────────────────────────────────────
# Figure 3: Per-class bar chart
# ─────────────────────────────────────────────────────────
valid_names = [n for n, a in per_class_acc.items() if a is not None]
valid_accs  = [per_class_acc[n] * 100 for n in valid_names]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#3b82f6" if a >= 50 else "#ef4444" for a in valid_accs]
bars   = ax.bar(valid_names, valid_accs, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(10, color="gray", linestyle="--", linewidth=1.2, label="Random (10%)")
ax.set_ylim(0, 115)
ax.set_ylabel("Test accuracy (%)", fontsize=10)
ax.set_title("SVM Baseline — Per-class test accuracy", fontsize=11)
ax.set_xticklabels(valid_names, rotation=35, ha="right", fontsize=9)
ax.legend(fontsize=9)
for bar, acc in zip(bars, valid_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{acc:.0f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("figures/baseline_per_class.png", dpi=150)
plt.close()
print("Saved figures/baseline_per_class.png")

# ─────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────
results = {
    "best_params":    gs.best_params_,
    "cv_accuracy":    round(gs.best_score_ * 100, 1),
    "val_accuracy":   round(val_acc * 100, 1),
    "test_accuracy":  round(test_acc * 100, 1),
    "per_class_acc":  {k: (round(v * 100, 1) if v is not None else None)
                       for k, v in per_class_acc.items()},
    "confusion_matrix": cm.tolist(),
    "label_names":    label_names,
    "n_train": int(tr_mask.sum()),
    "n_val":   int(val_mask.sum()),
    "n_test":  int(test_mask.sum()),
    "n_no_hand_detected": no_hand,
}
with open("results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*50)
print("BASELINE SUMMARY")
print("="*50)
print(f"  Best params:   {results['best_params']}")
print(f"  CV accuracy:   {results['cv_accuracy']}%")
print(f"  Val accuracy:  {results['val_accuracy']}%")
print(f"  Test accuracy: {results['test_accuracy']}%")
for name, acc in per_class_acc.items():
    s = f"{acc*100:.0f}%" if acc is not None else "N/A"
    print(f"  {name:15s}: {s}")
print("="*50)
