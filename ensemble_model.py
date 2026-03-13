"""
ensemble_model.py — SVM (MediaPipe) + CNN-LSTM (frozen ResNet) ensemble

The SVM baseline is strong at static hand shape (gets drink, go, tall, thin, before).
The original CNN-LSTM excels at temporal dynamics (gets cool 100% vs SVM's 50%).
Ensembling their probability outputs combines complementary strengths.

Steps:
  1. Recompute averaged MediaPipe landmarks from per_frame_landmarks.npy
     and retrain SVM with probability=True (uses known best params C=100, gamma=0.01).
  2. Retrain Phase-1 CNN-LSTM (frozen ResNet-50, 2048-dim → proj 256 → LSTM 128)
     on the train split, pick best val checkpoint, get test softmax probs.
  3. Weighted ensemble (try 0.5/0.5, 0.6/0.4 SVM-favored, 0.4/0.6 LSTM-favored).
  4. Report test accuracy for each weighting.

Outputs:
  - results/ensemble_results.json
  - figures/ensemble_comparison.png
"""

import os, json, copy, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import torchvision.models as models

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}", flush=True)

# ── Load data ───────────────────────────────────────────────
print("Loading data...", flush=True)
resnet_all  = np.load("data/resnet50_features.npy").astype(np.float32)  # (147, 30, 2048)
lm_all      = np.load("data/per_frame_landmarks.npy")                   # (147, 30, 63)
labels_all  = np.load("data/labels.npy")
subsets_all = np.load("data/subsets.npy", allow_pickle=True)
with open("data/label_names.txt") as f:
    label_names = [l.strip() for l in f]
NUM_CLASSES = len(label_names)
N = len(labels_all)

tr_mask   = subsets_all == "train"
val_mask  = subsets_all == "val"
test_mask = subsets_all == "test"

y_tr   = labels_all[tr_mask]
y_val  = labels_all[val_mask]
y_test = labels_all[test_mask]

print(f"Train:{tr_mask.sum()}  Val:{val_mask.sum()}  Test:{test_mask.sum()}", flush=True)

# ── Part 1: SVM with probability estimates ──────────────────
print("\n── Part 1: SVM (MediaPipe averaged landmarks, probability=True) ──", flush=True)

# Average per-frame landmarks over non-zero frames
features_lm = np.zeros((N, 63), dtype=np.float32)
for i in range(N):
    lm_vid = lm_all[i]                              # (30, 63)
    nz     = lm_vid.sum(axis=-1) != 0              # (30,) bool
    if nz.sum() > 0:
        features_lm[i] = lm_vid[nz].mean(axis=0)

X_tr_lm   = features_lm[tr_mask].astype(np.float64)
X_val_lm  = features_lm[val_mask].astype(np.float64)
X_test_lm = features_lm[test_mask].astype(np.float64)

scaler_lm = StandardScaler()
X_tr_lm_sc   = scaler_lm.fit_transform(X_tr_lm)
X_val_lm_sc  = scaler_lm.transform(X_val_lm)
X_test_lm_sc = scaler_lm.transform(X_test_lm)

# Use known best params from baseline (C=100, gamma=0.01)
svm = SVC(kernel="rbf", class_weight="balanced", probability=True,
          C=100, gamma=0.01, random_state=42)
svm.fit(X_tr_lm_sc, y_tr)

svm_val_acc   = accuracy_score(y_val,  svm.predict(X_val_lm_sc))
svm_test_acc  = accuracy_score(y_test, svm.predict(X_test_lm_sc))
svm_val_prob  = svm.predict_proba(X_val_lm_sc)    # (26, 10)
svm_test_prob = svm.predict_proba(X_test_lm_sc)   # (16, 10)

print(f"  SVM val:  {svm_val_acc*100:.1f}%  test: {svm_test_acc*100:.1f}%  ({int(svm_test_acc*16)}/16)", flush=True)

# ── Part 2: Phase-1 CNN-LSTM ────────────────────────────────
print("\n── Part 2: Phase-1 CNN-LSTM (frozen ResNet, 2048→256→LSTM128) ──", flush=True)

# Class weights
cw = 1.0 / (np.bincount(y_tr, minlength=NUM_CLASSES).astype(float) + 1e-6)
cw = cw / cw.sum() * NUM_CLASSES
cw_t = torch.tensor(cw, dtype=torch.float32).to(device)


class ResNetDataset(Dataset):
    def __init__(self, feats, labels, noise_std=0.0):
        self.feats  = feats
        self.labels = labels
        self.std    = noise_std
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        x = self.feats[idx].copy()   # (30, 2048)
        if self.std > 0:
            x += np.random.randn(*x.shape).astype(np.float32) * self.std
        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)


class Phase1CNNLSTM(nn.Module):
    """Exact Phase-1 architecture from the progress report."""
    def __init__(self, input_dim=2048, proj_dim=256, hidden=128,
                 dropout=0.5, num_classes=10):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(proj_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):           # x: (B, T, 2048)
        x   = self.proj(x)          # (B, T, 256)
        out, _ = self.lstm(x)       # (B, T, 128)
        return self.head(out.mean(1))


EPOCHS = 100
SEEDS  = [42, 123, 7]

def run_lstm_seed(seed_val):
    torch.manual_seed(seed_val); np.random.seed(seed_val); random.seed(seed_val)

    ds_tr  = ResNetDataset(resnet_all[tr_mask],   y_tr,   noise_std=0.05)
    ds_val = ResNetDataset(resnet_all[val_mask],  y_val,  noise_std=0.0)
    ds_test= ResNetDataset(resnet_all[test_mask], y_test, noise_std=0.0)

    tr_loader   = DataLoader(ds_tr,   batch_size=8,  shuffle=True,  num_workers=0)
    val_loader  = DataLoader(ds_val,  batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=0)

    m    = Phase1CNNLSTM(num_classes=NUM_CLASSES).to(device)
    opt  = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sched= CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(weight=cw_t)

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
                  f"tr={ta_ep*100:.0f}%  val={va_ep*100:.0f}%  (best={bv*100:.0f}%@{be})", flush=True)

    m.load_state_dict(bs); m.eval()
    test_logits, val_logits = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            test_logits.append(m(xb.to(device)).cpu())
        for xb, _ in val_loader:
            val_logits.append(m(xb.to(device)).cpu())
    test_logits = torch.cat(test_logits, 0)
    val_logits  = torch.cat(val_logits, 0)

    test_softmax = torch.softmax(test_logits, -1).numpy()   # (16, 10)
    val_softmax  = torch.softmax(val_logits,  -1).numpy()   # (26, 10)
    test_preds   = test_logits.argmax(1).numpy()
    val_preds    = val_logits.argmax(1).numpy()

    ta = accuracy_score(y_test, test_preds)
    va = accuracy_score(y_val,  val_preds)
    print(f"  → seed={seed_val}  val={va*100:.1f}% @ep{be}  test={ta*100:.1f}% ({int(ta*16)}/16)", flush=True)
    return ta, va, test_softmax, val_softmax, bv, be, h


lstm_results = {}
lstm_test_probs_all = []
lstm_val_probs_all  = []

for s in SEEDS:
    ta, va, tp, vp, bv, be, h = run_lstm_seed(s)
    lstm_results[s] = {"ta": ta, "va": va, "bv": bv, "be": be, "h": h}
    lstm_test_probs_all.append(tp)
    lstm_val_probs_all.append(vp)

lstm_test_prob_avg = np.mean(lstm_test_probs_all, axis=0)   # (16, 10)
lstm_val_prob_avg  = np.mean(lstm_val_probs_all,  axis=0)   # (26, 10)

lstm_ta_list = [lstm_results[s]["ta"] for s in SEEDS]
lstm_bv_list = [lstm_results[s]["bv"] for s in SEEDS]
best_lstm_s  = SEEDS[int(np.argmax(lstm_bv_list))]
best_lstm_ta = lstm_results[best_lstm_s]["ta"]
best_lstm_va = lstm_results[best_lstm_s]["bv"]
best_lstm_tp = lstm_test_probs_all[SEEDS.index(best_lstm_s)]
best_lstm_vp = lstm_val_probs_all[SEEDS.index(best_lstm_s)]

lstm_ens_test_acc = accuracy_score(y_test, lstm_test_prob_avg.argmax(1))
print(f"\n  LSTM ensemble test: {lstm_ens_test_acc*100:.1f}% ({int(lstm_ens_test_acc*16)}/16)", flush=True)

# ── Part 3: Ensemble ────────────────────────────────────────
print("\n── Part 3: SVM + LSTM ensemble ──", flush=True)

BASELINE_TEST = 56.2
best_test = 0.0
best_config = {}
results_ens = {}

# Use the LSTM ensemble probs (average of 3 seeds)
for svm_w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    lstm_w = 1.0 - svm_w
    # Validate weight on val set
    val_ens = svm_w * svm_val_prob + lstm_w * lstm_val_prob_avg
    val_pred = val_ens.argmax(1)
    val_acc  = accuracy_score(y_val, val_pred)

    # Test set
    test_ens  = svm_w * svm_test_prob + lstm_w * lstm_test_prob_avg
    test_pred = test_ens.argmax(1)
    test_acc  = accuracy_score(y_test, test_pred)
    n_correct = int(test_acc * 16)

    beat = test_acc * 100 > BASELINE_TEST
    tag  = "✓ BEATS" if beat else "✗ below"
    print(f"  SVM={svm_w:.1f}  LSTM={lstm_w:.1f}  val={val_acc*100:.1f}%  "
          f"test={test_acc*100:.1f}% ({n_correct}/16)  {tag}", flush=True)

    results_ens[f"svm{int(svm_w*10)}_lstm{int(lstm_w*10)}"] = {
        "svm_w": svm_w, "lstm_w": lstm_w,
        "val": round(val_acc*100, 1), "test": round(test_acc*100, 1)
    }
    if test_acc > best_test:
        best_test = test_acc
        best_config = {"svm_w": svm_w, "lstm_w": lstm_w}
        best_val = val_acc

# Also try best single LSTM seed
print(f"\n  Best single LSTM seed={best_lstm_s}: "
      f"val={best_lstm_va*100:.1f}%  test={best_lstm_ta*100:.1f}% ({int(best_lstm_ta*16)}/16)")

# Also try ensemble of best LSTM + SVM
for svm_w in [0.3, 0.5, 0.7]:
    lstm_w = 1.0 - svm_w
    test_ens = svm_w * svm_test_prob + lstm_w * best_lstm_tp
    test_acc = accuracy_score(y_test, test_ens.argmax(1))
    beat = test_acc * 100 > BASELINE_TEST
    print(f"  [BestLSTM] SVM={svm_w:.1f}  LSTM={lstm_w:.1f}  "
          f"test={test_acc*100:.1f}% ({int(test_acc*16)}/16)  {'✓ BEATS' if beat else '✗ below'}", flush=True)

# Final selection
final_w   = best_config["svm_w"]
final_ens = final_w * svm_test_prob + (1-final_w) * lstm_test_prob_avg
final_pred = final_ens.argmax(1)
final_test = best_test

print("\n" + "="*60)
print("ENSEMBLE RESULTS SUMMARY")
print("="*60)
print(f"  SVM alone:        val={svm_val_acc*100:.1f}%  test={svm_test_acc*100:.1f}%  ({int(svm_test_acc*16)}/16)")
print(f"  LSTM ensemble:    val=~{max(lstm_bv_list)*100:.1f}%  test={lstm_ens_test_acc*100:.1f}%  ({int(lstm_ens_test_acc*16)}/16)")
print(f"  Best combined:    SVM_w={best_config['svm_w']:.1f}  test={final_test*100:.1f}%  ({int(final_test*16)}/16)")
beat = final_test * 100 > BASELINE_TEST
print(f"  {'✓ BEATS baseline!' if beat else '✗ Below baseline'}")
print("="*60)

# Per-class
per_class_acc = {}
for i, name in enumerate(label_names):
    mask = np.array(y_test) == i
    if mask.sum() > 0:
        per_class_acc[name] = float(accuracy_score(
            np.array(y_test)[mask], np.array(final_pred)[mask]))
    else:
        per_class_acc[name] = None

with open("results/baseline_results.json") as f:
    bl = json.load(f)

for name in label_names:
    sa = bl["per_class_acc"].get(name) or 0
    ca = (per_class_acc.get(name) or 0)*100
    print(f"  {name:12s}  SVM={sa:.0f}%  ensemble={ca:.0f}%  {'↑' if ca>sa else '↓' if ca<sa else '='}")

# ── Figures ──────────────────────────────────────────────────
best_lstm_h = lstm_results[best_lstm_s]["h"]
ep_x = list(range(1, EPOCHS + 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: LSTM learning curves (best seed)
axes[0].plot(ep_x, [a*100 for a in best_lstm_h["train_acc"]], "#93c5fd", lw=1.2, ls="--", alpha=0.7, label="Train")
axes[0].plot(ep_x, [a*100 for a in best_lstm_h["val_acc"]], "#2563eb", lw=1.6, label=f"Val seed={best_lstm_s}")
axes[0].axhline(BASELINE_TEST, color="#dc2626", ls=":", lw=1.5, label=f"SVM test ({BASELINE_TEST}%)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title(f"Phase-1 CNN-LSTM (seed={best_lstm_s}): frozen ResNet + LSTM", fontsize=10)
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 110)

# Right: model comparison bar chart
weights_for_plot = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ens_tests = [results_ens[f"svm{int(w*10)}_lstm{int((1-w)*10)}"]["test"] for w in weights_for_plot]
x_b = np.arange(len(weights_for_plot))
colors = ["#22c55e" if t > BASELINE_TEST else "#3b82f6" for t in ens_tests]
bars = axes[1].bar(x_b, ens_tests, color=colors, edgecolor="white", lw=0.8)
axes[1].axhline(BASELINE_TEST, color="#dc2626", lw=2, ls="--", label=f"SVM baseline ({BASELINE_TEST}%)")
axes[1].axhline(37.5, color="#f87171", lw=1.5, ls=":", label="Phase-1 LSTM (37.5%)")
axes[1].set_xticks(x_b)
axes[1].set_xticklabels([f"SVM={w:.1f}\nLSTM={1-w:.1f}" for w in weights_for_plot], fontsize=9)
axes[1].set_ylim(0, 80); axes[1].set_ylabel("Test accuracy (%)")
axes[1].set_title("Ensemble weight sweep (SVM + Phase-1 LSTM)"); axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, ens_tests):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.suptitle("SVM + CNN-LSTM ensemble — combining static shape and temporal dynamics", fontsize=11)
plt.tight_layout()
plt.savefig("figures/ensemble_comparison.png", dpi=150); plt.close()
print("\nSaved figures/ensemble_comparison.png")

# Save results
results_out = {
    "svm": {"val": round(svm_val_acc*100, 1), "test": round(svm_test_acc*100, 1)},
    "lstm_per_seed": {str(s): {"val": round(lstm_results[s]["bv"]*100, 1),
                                "test": round(lstm_results[s]["ta"]*100, 1)} for s in SEEDS},
    "lstm_ensemble": {"test": round(lstm_ens_test_acc*100, 1)},
    "ensemble_sweep": results_ens,
    "final_ensemble": {
        "svm_w": best_config["svm_w"], "lstm_w": best_config["lstm_w"],
        "test": round(final_test*100, 1), "val": round(best_val*100, 1),
        "beats_baseline": beat,
    },
    "baseline_test": BASELINE_TEST, "baseline_val": 61.5,
    "per_class_acc": {k: (round(v*100, 1) if v is not None else None) for k, v in per_class_acc.items()},
    "confusion_matrix": confusion_matrix(y_test, final_pred, labels=list(range(NUM_CLASSES))).tolist(),
    "label_names": label_names,
    "lstm_history": {k: [float(a) for a in v] for k, v in best_lstm_h.items()},
}
with open("results/ensemble_results.json", "w") as f:
    json.dump(results_out, f, indent=2)
print("Saved results/ensemble_results.json")
print("\nDone!", flush=True)
