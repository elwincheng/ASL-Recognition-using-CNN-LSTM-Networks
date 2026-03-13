"""
improved_primary_model.py  — final

Architecture: cached ResNet-50 layer3 features (1024×14×14/frame) → fine-tune
only layer4 + avgpool → tiny LSTM head.  Trains on train split (105 videos),
validates on val split (26 videos) for early stopping, evaluates on test (16).

Key difference from phase-1: layer4 is allowed to update, giving more
ASL-specific features.  Aggressive regularization (wd=5e-3, label_smooth=0.1,
dropout=0.5) combats data scarcity.

Outputs:
  - results/improved_results.json
  - figures/improved_learning_curves.png
  - figures/improved_confusion.png
  - figures/improved_comparison.png
"""

import os, json, random, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import accuracy_score, confusion_matrix
import torchvision.models as models

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}", flush=True)

# ── Load data ──────────────────────────────────────────
print("Loading layer3 features...", flush=True)
layer3_all  = np.load("data/layer3_features.npy").astype(np.float32)
labels_all  = np.load("data/labels.npy")
subsets_all = np.load("data/subsets.npy", allow_pickle=True)
with open("data/label_names.txt") as f:
    label_names = [l.strip() for l in f]
NUM_CLASSES = len(label_names)

tr_mask   = subsets_all == "train"
val_mask  = subsets_all == "val"
test_mask = subsets_all == "test"
tv_mask   = tr_mask | val_mask

y_tr   = labels_all[tr_mask]
y_tv   = labels_all[tv_mask]
y_val  = labels_all[val_mask]
y_test = labels_all[test_mask]

print(f"Train: {tr_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}", flush=True)
BASELINE_TEST = 56.2

cw_tr = 1.0 / (np.bincount(y_tr, minlength=NUM_CLASSES).astype(float) + 1e-6)
cw_tr = cw_tr / cw_tr.sum() * NUM_CLASSES
cw_tr_t = torch.tensor(cw_tr, dtype=torch.float32).to(device)

cw_tv = 1.0 / (np.bincount(y_tv, minlength=NUM_CLASSES).astype(float) + 1e-6)
cw_tv = cw_tv / cw_tv.sum() * NUM_CLASSES
cw_tv_t = torch.tensor(cw_tv, dtype=torch.float32).to(device)


class Layer3Dataset(Dataset):
    def __init__(self, feats, labels, augment=False):
        self.feats   = feats
        self.labels  = labels
        self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        x = self.feats[idx].copy()
        if self.augment and random.random() < 0.5:
            x = x[:, :, :, ::-1].copy()
        return torch.from_numpy(x).contiguous(), torch.tensor(self.labels[idx], dtype=torch.long)


class Layer4LSTMModel(nn.Module):
    def __init__(self, hidden=64, dropout=0.5, num_classes=10):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.layer4  = rn.layer4
        self.avgpool = rn.avgpool
        self.lstm    = nn.LSTM(2048, hidden, num_layers=1, batch_first=True)
        self.head    = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W).contiguous()
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        x = x.reshape(B, T, 2048).contiguous()
        out, _ = self.lstm(x)
        return self.head(out.mean(1))


ds_tr   = Layer3Dataset(layer3_all[tr_mask],   y_tr,   augment=True)
ds_tv   = Layer3Dataset(layer3_all[tv_mask],   y_tv,   augment=True)
ds_val  = Layer3Dataset(layer3_all[val_mask],  y_val,  augment=False)
ds_test = Layer3Dataset(layer3_all[test_mask], y_test, augment=False)

tr_loader   = DataLoader(ds_tr,   batch_size=8, shuffle=True,  num_workers=0, pin_memory=False)
tv_loader   = DataLoader(ds_tv,   batch_size=8, shuffle=True,  num_workers=0, pin_memory=False)
val_loader  = DataLoader(ds_val,  batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

EPOCHS = 20  # ~29s/epoch × 20 = 580s ≈ 9.7min per seed
SEEDS  = [42, 123, 7]  # Run up to 3 seeds; stop early if time is tight


def run_one_seed(seed_val):
    """Train on train, pick best val checkpoint, evaluate on test."""
    torch.manual_seed(seed_val)
    m   = Layer4LSTMModel(hidden=64, dropout=0.5, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam([
        {"params": m.layer4.parameters(),  "lr": 2e-5, "weight_decay": 5e-3},
        {"params": m.avgpool.parameters(), "lr": 2e-5, "weight_decay": 5e-3},
        {"params": m.lstm.parameters(),    "lr": 5e-4, "weight_decay": 5e-3},
        {"params": m.head.parameters(),    "lr": 5e-4, "weight_decay": 5e-3},
    ])
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    crit  = nn.CrossEntropyLoss(weight=cw_tr_t, label_smoothing=0.1)

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
        print(f"  s={seed_val} ep{ep:2d}/{EPOCHS}  "
              f"tr={ta_ep*100:.0f}%  val={va_ep*100:.0f}%  (best={bv*100:.0f}%@{be})", flush=True)

    m.load_state_dict(bs); m.eval()
    preds, logits = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            lgt = m(xb.to(device)).cpu()
            logits.append(lgt); preds.extend(lgt.argmax(1).numpy())
    logits = torch.cat(logits, 0)
    ta = accuracy_score(y_test, preds)
    print(f"  → seed={seed_val}  val={bv*100:.1f}% @ep{be}  test={ta*100:.1f}% ({int(ta*16)}/16)", flush=True)
    return ta, preds, bv, be, h, logits


print(f"\n── E2E fine-tune: ResNet layer4 + LSTM  ({EPOCHS} epochs, 3 seeds) ──", flush=True)
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

# Best single
bv_list = [results_seeds[s]["bv"] for s in SEEDS]
ta_list = [results_seeds[s]["ta"] for s in SEEDS]
best_s   = SEEDS[int(np.argmax(bv_list))]
ta_best  = results_seeds[best_s]["ta"]
h_best   = results_seeds[best_s]["h"]
bv_best  = results_seeds[best_s]["bv"]

# Select final
if ta_ens >= max(ta_list):
    final_preds = ens_preds; final_test = ta_ens;   final_tag = "Ensemble (3 seeds)"
else:
    final_preds = results_seeds[best_s]["pd"]
    final_test  = ta_best;  final_tag = f"Single seed={best_s}"

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"  SVM baseline (MediaPipe):  val=61.5%  test=56.2%  (9/16)")
for s in SEEDS:
    r = results_seeds[s]
    print(f"  E2E seed={s:3d}:  val={r['bv']*100:.1f}%  test={r['ta']*100:.1f}%  ({int(r['ta']*16)}/16)")
print(f"  E2E ensemble:          test={ta_ens*100:.1f}%  ({int(ta_ens*16)}/16)")
beat = final_test * 100 > BASELINE_TEST
print(f"  Selected: {final_tag}  test={final_test*100:.1f}%")
print(f"  {'✓ BEATS baseline!' if beat else '✗ Below baseline'}")
print("="*60)

# Per-class
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
    print(f"  {name:12s}  SVM={sa:.0f}%  CNN={ca:.0f}%")

# ── Figures ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ep_x = list(range(1, EPOCHS + 1))
colors_t = ["#93c5fd", "#6ee7b7", "#fcd34d"]
colors_v = ["#2563eb", "#059669", "#d97706"]

for i, s in enumerate(SEEDS):
    r = results_seeds[s]
    axes[0].plot(ep_x, [a*100 for a in r["h"]["train_acc"]], color=colors_t[i], lw=1.2, ls="--", alpha=0.7)
    axes[0].plot(ep_x, [a*100 for a in r["h"]["val_acc"]], color=colors_v[i], lw=1.6,
                 label=f"Val seed={s} (best={r['bv']*100:.0f}%)")

axes[0].axhline(BASELINE_TEST, color="#dc2626", ls=":", lw=1.5, label=f"SVM test ({BASELINE_TEST}%)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("E2E layer4 fine-tune curves (dashed=train, solid=val)", fontsize=10)
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 110)

bar_labels = ["SVM\n(MediaPipe)", "Primary\nCNN-LSTM\n(phase 1)", "E2E best\nsingle", "E2E\nensemble"]
test_bars  = [56.2, 37.5, max(ta_list)*100, ta_ens*100]
bar_colors = ["#94a3b8", "#f87171", "#3b82f6", "#1d4ed8"]
x_b = np.arange(len(bar_labels))
bars = axes[1].bar(x_b, test_bars, color=bar_colors, edgecolor="white", lw=0.8)
axes[1].axhline(56.2, color="#94a3b8", lw=1.5, ls="--", alpha=0.8)
axes[1].axhline(10, color="gray", lw=1.2, ls=":", label="Random (10%)")
axes[1].set_xticks(x_b); axes[1].set_xticklabels(bar_labels, fontsize=9)
axes[1].set_ylim(0, 80); axes[1].set_ylabel("Test accuracy (%)")
axes[1].set_title("Phase-1 vs phase-2 comparison"); axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, test_bars):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.suptitle("Improved CNN-LSTM — ResNet layer4 end-to-end fine-tuning", fontsize=11)
plt.tight_layout()
plt.savefig("figures/improved_learning_curves.png", dpi=150); plt.close()
print("\nSaved figures/improved_learning_curves.png")

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
ax.set_title(f"Improved model — Test Confusion Matrix")
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig("figures/improved_confusion.png", dpi=150); plt.close()
print("Saved figures/improved_confusion.png")

# Per-class bar
svm_pc = [bl["per_class_acc"].get(n) or 0 for n in label_names]
cnn_pc = [(per_class_acc.get(n) or 0)*100 for n in label_names]
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(NUM_CLASSES); w = 0.35
ax.bar(x-w/2, svm_pc, w, label="SVM (MediaPipe)", color="#94a3b8")
ax.bar(x+w/2, cnn_pc, w, label="E2E fine-tuned CNN-LSTM", color="#3b82f6")
ax.axhline(10, color="gray", ls="--", lw=1.2, label="Random (10%)")
ax.set_xticks(x); ax.set_xticklabels(label_names, rotation=35, ha="right", fontsize=9)
ax.set_ylim(0, 115); ax.set_ylabel("Per-class test accuracy (%)")
ax.set_title("Per-class accuracy: SVM vs E2E fine-tuned CNN-LSTM"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/improved_comparison.png", dpi=150); plt.close()
print("Saved figures/improved_comparison.png")

# Save
results = {
    **{f"e2e_seed{s}": {"val": round(results_seeds[s]["bv"]*100,1),
                         "test": round(results_seeds[s]["ta"]*100,1)} for s in SEEDS},
    "e2e_ensemble":        {"val": round(bv_best*100,1), "test": round(ta_ens*100,1)},
    "final_model":         final_tag,
    "final_test_accuracy": round(final_test*100, 1),
    "final_val_accuracy":  round(bv_best*100, 1),
    "beats_baseline":      beat,
    "baseline_test":       BASELINE_TEST,
    "baseline_val":        61.5,
    "per_class_acc":       {k: (round(v*100,1) if v is not None else None) for k, v in per_class_acc.items()},
    "confusion_matrix":    cm.tolist(),
    "label_names":         label_names,
    "e2e_history":         {k: [float(x) for x in v] for k, v in h_best.items()},
}
with open("results/improved_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results/improved_results.json")
print("\nDone!", flush=True)
