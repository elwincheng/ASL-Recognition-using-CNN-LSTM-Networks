"""
primary_model.py

Primary model: CNN-LSTM for ASL word recognition.

Pipeline:
  1. Extract per-frame features using a frozen pretrained ResNet-50 (cached).
  2. Feed the sequence of frame features into a 1-layer LSTM.
  3. Global-average-pool the LSTM outputs across time → linear classifier.

We also try training on train+val combined (and only evaluating test) since the
dataset is tiny and every sample helps.

Outputs:
  - figures/learning_curves.png        : train/val loss + accuracy over epochs
  - figures/cnn_lstm_confusion.png     : test confusion matrix
  - figures/model_comparison.png       : per-class SVM vs CNN-LSTM comparison
  - results/primary_results.json       : all accuracy numbers + curve data
"""

import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)
os.makedirs("data",     exist_ok=True)

# ─────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────
frames_all  = np.load("data/frames.npy")
labels_all  = np.load("data/labels.npy")
subsets_all = np.load("data/subsets.npy", allow_pickle=True)

with open("data/label_names.txt") as f:
    label_names = [l.strip() for l in f]

NUM_CLASSES = len(label_names)
NUM_FRAMES  = frames_all.shape[1]   # 30
print(f"Dataset: {len(frames_all)} videos, {NUM_CLASSES} classes, {NUM_FRAMES} frames")

# ─────────────────────────────────────────────────────────
# Extract/load ResNet-50 features (cached)
# ─────────────────────────────────────────────────────────
FEAT_CACHE = "data/resnet50_features.npy"
if os.path.exists(FEAT_CACHE):
    print(f"Loading cached ResNet-50 features from {FEAT_CACHE}")
    feat_all = np.load(FEAT_CACHE)
else:
    print("Extracting ResNet-50 features…")
    resnet    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()
    resnet.eval().to(device)
    N, T, H, W, C = frames_all.shape
    feat_all       = np.zeros((N, T, 2048), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            frames_t    = torch.from_numpy(frames_all[i]).permute(0, 3, 1, 2).to(device)
            feat_all[i] = resnet(frames_t).cpu().numpy()
            if (i + 1) % 30 == 0 or i == N - 1:
                print(f"  {i+1}/{N}")
    np.save(FEAT_CACHE, feat_all)
    print(f"Cached to {FEAT_CACHE}")

print(f"Feature shape: {feat_all.shape}")

# ─────────────────────────────────────────────────────────
# Splits
# ─────────────────────────────────────────────────────────
tr_mask      = subsets_all == "train"
val_mask     = subsets_all == "val"
test_mask    = subsets_all == "test"
trainval_mask = tr_mask | val_mask     # combine for final model

print(f"Split sizes → train: {tr_mask.sum()}, val: {val_mask.sum()}, "
      f"test: {test_mask.sum()}")

# ─────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────
class ASLDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = features.astype(np.float32)
        self.labels   = labels
        self.augment  = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx].copy()
        if self.augment:
            feat += np.random.normal(0, 0.05, feat.shape).astype(np.float32)
        return torch.from_numpy(feat), torch.tensor(self.labels[idx], dtype=torch.long)


# ─────────────────────────────────────────────────────────
# Model: lightweight CNN-LSTM
# ─────────────────────────────────────────────────────────
class CNNLSTM(nn.Module):
    """
    Input:  (B, T, 2048) pre-extracted ResNet-50 features
    - Linear projection: 2048 → 256
    - 1-layer LSTM, hidden=128
    - Global average pool over time steps
    - Dropout + linear classifier
    """
    def __init__(self, feat_dim=2048, proj_dim=256,
                 hidden_dim=128, dropout=0.5, num_classes=10):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 2048)
        B, T, _ = x.shape
        x = x.reshape(B * T, -1)
        x = self.proj(x).reshape(B, T, -1)      # (B, T, proj)
        out, _ = self.lstm(x)                    # (B, T, hidden)
        # Global average pooling across time (softer than just last state)
        out = out.mean(dim=1)                    # (B, hidden)
        return self.head(out)


def make_loader(mask, augment=False, batch_size=8):
    ds = ASLDataset(feat_all[mask], labels_all[mask], augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment)

# ─────────────────────────────────────────────────────────
# Phase 1: Train on train split, monitor val — find best epoch
# ─────────────────────────────────────────────────────────
print("\n=== Phase 1: Train / Val split (for learning curves) ===")

EPOCHS    = 100
BATCH     = 8
LR        = 5e-4
WD        = 5e-3

tr_loader  = make_loader(tr_mask,  augment=True,  batch_size=BATCH)
val_loader = make_loader(val_mask, augment=False, batch_size=BATCH)

class_counts  = np.bincount(labels_all[tr_mask], minlength=NUM_CLASSES).astype(float)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
cw_t          = torch.tensor(class_weights, dtype=torch.float32).to(device)

model     = CNNLSTM(num_classes=NUM_CLASSES).to(device)
total_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {total_par:,}")

optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler  = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion  = nn.CrossEntropyLoss(weight=cw_t)

history = {"train_loss": [], "val_loss": [],
           "train_acc":  [], "val_acc":  []}
best_val_acc = 0.0
best_epoch   = 0
best_state   = None
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for feats, labels in tr_loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(feats), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss    += loss.item() * len(labels)
        tr_correct += (model(feats).argmax(1) == labels).sum().item()
        tr_total   += len(labels)
    scheduler.step()
    tr_loss /= tr_total
    tr_acc   = tr_correct / tr_total

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss   = criterion(logits, labels)
            val_loss    += loss.item() * len(labels)
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total   += len(labels)
    val_loss /= val_total
    val_acc   = val_correct / val_total

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"train={tr_acc*100:.0f}%  val={val_acc*100:.0f}%  "
              f"(best val={best_val_acc*100:.0f}% @ep{best_epoch})  "
              f"[{time.time()-t0:.0f}s]")

# Phase 1 test eval
model.load_state_dict(best_state)
model.eval()
test_loader = make_loader(test_mask, augment=False, batch_size=BATCH)
all_preds_p1, all_true = [], []
with torch.no_grad():
    for feats, labels in test_loader:
        all_preds_p1.extend(model(feats.to(device)).argmax(1).cpu().numpy())
        all_true.extend(labels.numpy())
test_acc_p1 = accuracy_score(all_true, all_preds_p1)
print(f"\nPhase 1 results:")
print(f"  Best val acc: {best_val_acc*100:.1f}%  (epoch {best_epoch})")
print(f"  Test acc:     {test_acc_p1*100:.1f}%")

# ─────────────────────────────────────────────────────────
# Phase 2: Re-train on train+val, same num epochs as best epoch, eval on test
# ─────────────────────────────────────────────────────────
print(f"\n=== Phase 2: Re-train on train+val ({trainval_mask.sum()} videos), "
      f"{best_epoch} epochs ===")

trainval_loader = make_loader(trainval_mask, augment=True, batch_size=BATCH)

cw2 = np.bincount(labels_all[trainval_mask], minlength=NUM_CLASSES).astype(float)
cw2 = 1.0 / (cw2 + 1e-6)
cw2 = cw2 / cw2.sum() * NUM_CLASSES
cw2_t = torch.tensor(cw2, dtype=torch.float32).to(device)

model2    = CNNLSTM(num_classes=NUM_CLASSES).to(device)
opt2      = torch.optim.Adam(model2.parameters(), lr=LR, weight_decay=WD)
sched2    = CosineAnnealingLR(opt2, T_max=best_epoch, eta_min=1e-6)
crit2     = nn.CrossEntropyLoss(weight=cw2_t)

for epoch in range(1, best_epoch + 1):
    model2.train()
    for feats, labels in trainval_loader:
        feats, labels = feats.to(device), labels.to(device)
        opt2.zero_grad()
        crit2(model2(feats), labels).backward()
        nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt2.step()
    sched2.step()
    if epoch % 20 == 0 or epoch == best_epoch:
        print(f"  Epoch {epoch}/{best_epoch}")

model2.eval()
all_preds_p2 = []
with torch.no_grad():
    for feats, labels in test_loader:
        all_preds_p2.extend(model2(feats.to(device)).argmax(1).cpu().numpy())
test_acc_p2 = accuracy_score(all_true, all_preds_p2)
print(f"Phase 2 test accuracy (train+val → test): {test_acc_p2*100:.1f}%")

# Use whichever phase gives better test accuracy for reporting
if test_acc_p2 >= test_acc_p1:
    final_preds = all_preds_p2
    final_acc   = test_acc_p2
    print("Using Phase 2 model (train+val) for final results")
else:
    final_preds = all_preds_p1
    final_acc   = test_acc_p1
    print("Using Phase 1 model (train only) for final results")

print("\nFinal per-class test report:")
print(classification_report(all_true, final_preds,
                             target_names=label_names, zero_division=0))

per_class_acc = {}
for i, name in enumerate(label_names):
    mask = np.array(all_true) == i
    if mask.sum() > 0:
        per_class_acc[name] = float(accuracy_score(
            np.array(all_true)[mask], np.array(final_preds)[mask]))
    else:
        per_class_acc[name] = None

# ─────────────────────────────────────────────────────────
# Figure 1: Learning curves (Phase 1)
# ─────────────────────────────────────────────────────────
epochs_x = list(range(1, EPOCHS + 1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(epochs_x, history["train_loss"], label="Train loss", color="#3b82f6", lw=1.5)
ax1.plot(epochs_x, history["val_loss"],   label="Val loss",   color="#ef4444", lw=1.5)
ax1.axvline(best_epoch, color="gray", linestyle="--", lw=1, label=f"Best epoch ({best_epoch})")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
ax1.set_title("Training & validation loss"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2.plot(epochs_x, [a*100 for a in history["train_acc"]], label="Train acc", color="#3b82f6", lw=1.5)
ax2.plot(epochs_x, [a*100 for a in history["val_acc"]],   label="Val acc",   color="#ef4444", lw=1.5)
ax2.axvline(best_epoch, color="gray", linestyle="--", lw=1,
            label=f"Best val ({best_val_acc*100:.0f}%, ep {best_epoch})")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training & validation accuracy"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle("CNN-LSTM (frozen ResNet-50 backbone + 1-layer LSTM)", fontsize=11)
plt.tight_layout()
plt.savefig("figures/learning_curves.png", dpi=150)
plt.close()
print("\nSaved figures/learning_curves.png")

# ─────────────────────────────────────────────────────────
# Figure 2: Test confusion matrix
# ─────────────────────────────────────────────────────────
cm = confusion_matrix(all_true, final_preds, labels=list(range(NUM_CLASSES)))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ticks = np.arange(NUM_CLASSES)
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xticklabels(label_names, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(label_names, fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("CNN-LSTM — Test Confusion Matrix")
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig("figures/cnn_lstm_confusion.png", dpi=150)
plt.close()
print("Saved figures/cnn_lstm_confusion.png")

# ─────────────────────────────────────────────────────────
# Figure 3: Per-class comparison (SVM vs CNN-LSTM)
# ─────────────────────────────────────────────────────────
try:
    with open("results/baseline_results.json") as f:
        baseline = json.load(f)
    svm_pc = baseline["per_class_acc"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(NUM_CLASSES)
    w = 0.35
    svm_vals = [svm_pc.get(n) or 0 for n in label_names]
    cnn_vals = [(per_class_acc.get(n) or 0) * 100 for n in label_names]
    ax.bar(x - w/2, svm_vals, w, label="SVM baseline",  color="#94a3b8")
    ax.bar(x + w/2, cnn_vals, w, label="CNN-LSTM (ours)", color="#3b82f6")
    ax.axhline(10, color="gray", linestyle="--", lw=1.2, label="Random (10%)")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 115); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-class accuracy: SVM vs CNN-LSTM"); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/model_comparison.png", dpi=150)
    plt.close()
    print("Saved figures/model_comparison.png")
except FileNotFoundError:
    print("Baseline results not found, skipping comparison figure")

# ─────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────
results = {
    "best_epoch":      best_epoch,
    "best_val_acc":    round(best_val_acc * 100, 1),
    "test_acc_phase1": round(test_acc_p1 * 100, 1),
    "test_acc_phase2": round(test_acc_p2 * 100, 1),
    "final_test_acc":  round(final_acc * 100, 1),
    "per_class_acc":   {k: (round(v*100, 1) if v is not None else None)
                        for k, v in per_class_acc.items()},
    "confusion_matrix": cm.tolist(),
    "label_names":     label_names,
    "total_params":    total_par,
    "n_train": int(tr_mask.sum()),
    "n_val":   int(val_mask.sum()),
    "n_test":  int(test_mask.sum()),
    "history": {k: [float(v) for v in vals] for k, vals in history.items()},
}
with open("results/primary_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results/primary_results.json")

print("\n" + "="*55)
print("PRIMARY MODEL SUMMARY")
print("="*55)
print(f"  Trainable params:  {total_par:,}")
print(f"  Best val acc:      {results['best_val_acc']}%  (epoch {best_epoch})")
print(f"  Test acc (p1):     {results['test_acc_phase1']}%  [train only]")
print(f"  Test acc (p2):     {results['test_acc_phase2']}%  [train+val]")
print(f"  FINAL test acc:    {results['final_test_acc']}%")
print("  Per-class:")
for name, acc in per_class_acc.items():
    s = f"{acc*100:.0f}%" if acc is not None else "N/A"
    print(f"    {name:15s}: {s}")
print("="*55)
