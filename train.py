"""
SST-5 Sentiment Classifier — Improved Training Pipeline v2
Model   : roberta-base  (+3-4% vs DistilBERT on SST-5)
Upgrades:
  • roberta-base              — stronger contextual representations
  • MAX_LEN 256               — captures full sentence context
  • label_smoothing_factor 0.1 — reduces overconfidence on hard boundary classes
  • warmup_ratio 0.1          — stable early training, less loss spiking
  • LR 1e-5                   — lower LR suits larger model
  • Batch 16 + grad_accum 2  → effective batch 32, fits in Mac RAM
  • Early stopping patience 3 → gives model more time to recover
"""

import json
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import transformers as _tf
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME      = "roberta-base"      # ← upgraded from distilbert-base-uncased
OUTPUT_DIR      = "./sst5_model_v2"
RESULTS_DIR     = "./results_v2"
MAX_LEN         = 256                 # ← up from 128; captures more context
BATCH_SIZE      = 16                  # smaller per-device; grad_accum compensates
GRAD_ACCUM      = 2                   # effective batch = 16 × 2 = 32
EPOCHS          = 6
LR              = 1e-5                # ← lower suits larger model
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1                 # ← 10 % of steps for LR warm-up
LABEL_SMOOTHING = 0.1                 # ← softens hard labels → better neutral F1
SEED            = 42

LABEL_NAMES = ["very negative", "negative", "neutral", "positive", "very positive"]
NUM_LABELS  = len(LABEL_NAMES)
id2label    = {i: l for i, l in enumerate(LABEL_NAMES)}
label2id    = {l: i for i, l in enumerate(LABEL_NAMES)}

Path(RESULTS_DIR).mkdir(exist_ok=True)

# ── 1. Dataset ───────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading dataset …")
print("=" * 60)
dataset = load_dataset("SetFit/sst5")
print(dataset)

# ── 2. Tokeniser ─────────────────────────────────────────────────────────────
print("\n2. Loading tokeniser …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text", "label_text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ── 3. Model ─────────────────────────────────────────────────────────────────
print("\n3. Loading model …")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {total_params:,}  (~{total_params/1e6:.0f}M)")

# ── 4. Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

# ── 5. Training Arguments ────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    label_smoothing_factor=LABEL_SMOOTHING,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    seed=SEED,
    report_to="none",
    fp16=torch.cuda.is_available(),   # CUDA only — False on Mac MPS, safe
)

# ── 6. Trainer  (handles tokenizer/processing_class rename across versions) ──
_trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
_params = inspect.signature(_tf.Trainer.__init__).parameters
_trainer_kwargs["processing_class" if "processing_class" in _params else "tokenizer"] = tokenizer

trainer = Trainer(**_trainer_kwargs)

# ── 7. Train ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Training …  (RoBERTa is ~2× slower than DistilBERT — worth it!)")
print("=" * 60)
train_result = trainer.train()
print(f"\nTraining done  ✓  ({train_result.metrics['train_runtime']:.1f}s)")

# ── 8. Evaluate ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Evaluating on test split …")
print("=" * 60)
predictions = trainer.predict(tokenized["test"])
preds  = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

report = classification_report(labels, preds, target_names=LABEL_NAMES, digits=4)
print(report)

metrics = {
    "accuracy":    float(accuracy_score(labels, preds)),
    "f1_macro":    float(f1_score(labels, preds, average="macro")),
    "f1_weighted": float(f1_score(labels, preds, average="weighted")),
}
print("Summary:", json.dumps(metrics, indent=2))

with open(f"{RESULTS_DIR}/test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
    f.write(report)

# ── 9. Save ───────────────────────────────────────────────────────────────────
print("\n6. Saving model …")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   Saved to {OUTPUT_DIR}/")

# ── 10. Visualisations ────────────────────────────────────────────────────────
print("\n7. Generating visualisations …")

COLORS = {
    "bg":   "#0d1117", "panel": "#161b22",
    "a1":   "#f78166", "a2":   "#3fb950",
    "a3":   "#58a6ff", "text": "#e6edf3",
    "grid": "#21262d",
}
LABEL_COLORS = ["#f78166", "#ffa657", "#b3b3f0", "#3fb950", "#58a6ff"]

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],    "axes.facecolor": COLORS["panel"],
    "text.color":       COLORS["text"],  "axes.labelcolor": COLORS["text"],
    "xtick.color":      COLORS["text"],  "ytick.color":     COLORS["text"],
    "axes.edgecolor":   COLORS["grid"],  "grid.color":      COLORS["grid"],
    "font.family":      "monospace",
})

# — Training history ——————————————————————————————————————————————————————————
log_history = trainer.state.log_history
train_loss, eval_loss, eval_acc, eval_f1 = [], [], [], []
for log in log_history:
    if "loss" in log and "eval_loss" not in log:
        train_loss.append((log["step"], log["loss"]))
    if "eval_loss" in log:
        eval_loss.append((log["epoch"], log["eval_loss"]))
        eval_acc.append((log["epoch"], log["eval_accuracy"]))
        eval_f1.append((log["epoch"], log["eval_f1_macro"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.patch.set_facecolor(COLORS["bg"])
fig.suptitle("Training History — SST-5 RoBERTa-base v2",
             color=COLORS["text"], fontsize=13, fontweight="bold", y=1.02)

ax = axes[0]
if train_loss:
    xs, ys = zip(*train_loss)
    ax.plot(xs, ys, color=COLORS["a1"], lw=1.5, alpha=0.85, label="train loss")
if eval_loss:
    xs, ys = zip(*eval_loss)
    ax.plot(xs, ys, color=COLORS["a3"], lw=2.5, marker="o", label="val loss")
ax.set_title("Loss", color=COLORS["text"])
ax.set_xlabel("step / epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

ax = axes[1]
if eval_acc:
    xs, ys = zip(*eval_acc)
    ax.plot(xs, ys, color=COLORS["a2"], lw=2.5, marker="o")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("Validation Accuracy", color=COLORS["text"])
ax.set_xlabel("epoch"); ax.grid(True, alpha=0.25)

ax = axes[2]
if eval_f1:
    xs, ys = zip(*eval_f1)
    ax.plot(xs, ys, color=COLORS["a3"], lw=2.5, marker="o")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("Validation F1 (macro)", color=COLORS["text"])
ax.set_xlabel("epoch"); ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_history.png", dpi=150,
            bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print("   ✓ training_history.png")

# — Confusion matrix ——————————————————————————————————————————————————————————
cm      = confusion_matrix(labels, preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(COLORS["bg"])
fig.suptitle("Confusion Matrix — SST-5 Test Set (RoBERTa v2)",
             color=COLORS["text"], fontsize=12, fontweight="bold")

short = ["V.Neg", "Neg", "Neu", "Pos", "V.Pos"]
for ax, data, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ["Raw counts", "Normalised (row %)"],
    ["d", ".2f"],
):
    sns.heatmap(
        data, ax=ax, annot=True, fmt=fmt, cmap="mako",
        xticklabels=short, yticklabels=short,
        linewidths=0.5, linecolor=COLORS["bg"],
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10, "color": "white"},
    )
    ax.set_title(title, color=COLORS["text"])
    ax.set_xlabel("Predicted", color=COLORS["text"])
    ax.set_ylabel("Actual",    color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150,
            bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print("   ✓ confusion_matrix.png")

# — Per-class F1 ——————————————————————————————————————————————————————————————
per_class_f1 = f1_score(labels, preds, average=None)

fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(COLORS["bg"])
bars = ax.barh(LABEL_NAMES, per_class_f1,
               color=LABEL_COLORS, edgecolor="none", height=0.55)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlim(0, 1.08)
ax.set_title("Per-class F1 Score — Test Set (RoBERTa v2)",
             color=COLORS["text"], fontsize=12, fontweight="bold")
ax.grid(True, axis="x", alpha=0.25)
for bar, val in zip(bars, per_class_f1):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", color=COLORS["text"], fontsize=9)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/per_class_f1.png", dpi=150,
            bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print("   ✓ per_class_f1.png")

# — v1 vs v2 comparison (auto-renders if old results/test_metrics.json exists) —
v1_path = Path("./results/test_metrics.json")
if v1_path.exists():
    with open(v1_path) as f:
        v1 = json.load(f)
    v2 = metrics

    metric_keys   = ["accuracy", "f1_macro", "f1_weighted"]
    metric_labels = ["Accuracy", "F1 Macro", "F1 Weighted"]
    v1_vals = [v1[k] for k in metric_keys]
    v2_vals = [v2[k] for k in metric_keys]

    x = np.arange(len(metric_keys))
    w = 0.32
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(COLORS["bg"])
    b1 = ax.bar(x - w/2, v1_vals, w, label="v1  DistilBERT",
                color=COLORS["a1"], alpha=0.85, edgecolor="none")
    b2 = ax.bar(x + w/2, v2_vals, w, label="v2  RoBERTa",
                color=COLORS["a2"], alpha=0.85, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0.40, 0.75)
    ax.set_title("v1 DistilBERT vs v2 RoBERTa — Test Metrics",
                 color=COLORS["text"], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height()*100:.1f}%",
                ha="center", color=COLORS["text"], fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/v1_vs_v2_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print("   ✓ v1_vs_v2_comparison.png")
else:
    print("   (skipping comparison chart — no v1 results found)")

print("\n" + "=" * 60)
print("Pipeline complete ✓")
print(f"  Model   → {OUTPUT_DIR}/")
print(f"  Results → {RESULTS_DIR}/")
print(f"  Accuracy     : {metrics['accuracy']:.4f}")
print(f"  F1 (macro)   : {metrics['f1_macro']:.4f}")
print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
print("=" * 60)
