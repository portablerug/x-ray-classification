import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, precision_recall_curve, average_precision_score,
)

# ── Paths ────────────────────────────────────────────────────────────────────
# On Hugging Face all files sit next to app.py in /app/
# Locally, model files live in project_root/models/ (two levels up from gradio/)
_here = Path(__file__).resolve().parent
if (_here / "best_threshold.txt").exists():
    MODELS_DIR = _here
else:
    MODELS_DIR = _here.parent.parent / "models"

MODEL_PATH  = MODELS_DIR / "resnet50_phase5.pt"
THRESH_PATH = MODELS_DIR / "best_threshold.txt"
PREDS_PATH  = MODELS_DIR / "predictions_resnet50.csv"

THRESHOLD = float(THRESH_PATH.read_text().strip())

# ── Load model once at startup ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Load saved test predictions once at startup ───────────────────────────────
df_preds   = pd.read_csv(PREDS_PATH)
all_labels = df_preds["true_label"].values
all_probs  = df_preds["prob_disease"].values
all_preds  = (all_probs >= THRESHOLD).astype(int)

# ── Pre-generate performance figures ─────────────────────────────────────────
def make_performance_figures():
    auc       = roc_auc_score(all_labels, all_probs)
    acc       = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)

    # ── Figure 1: ROC + Precision-Recall ─────────────────────────────────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    RocCurveDisplay.from_predictions(all_labels, all_probs,
                                     name=f"ResNet-50  (AUC={auc:.3f})", ax=ax1)
    ax1.plot([0, 1], [0, 1], "k--", label="Random")
    ax1.set_title("ROC Curve")
    ax1.legend(fontsize=9)

    pc, rc, _ = precision_recall_curve(all_labels, all_probs)
    ap        = average_precision_score(all_labels, all_probs)
    prev      = all_labels.mean()
    ax2.plot(rc, pc, color="darkorange", label=f"ResNet-50  (AP={ap:.3f})")
    ax2.axhline(y=prev, color="gray", linestyle="--", label=f"Random  (AP={prev:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)

    fig1.suptitle("Model Evaluation Curves", fontsize=13)
    fig1.tight_layout()

    # ── Figure 2: Confusion matrix ────────────────────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm, display_labels=["No Finding", "Disease"]).plot(
        ax=ax3, colorbar=False, cmap="Blues")
    ax3.set_title(f"Confusion Matrix  (threshold={THRESHOLD:.2f})")
    fig2.tight_layout()

    # ── Metrics summary text ──────────────────────────────────────────────────
    summary = (
        f"### Model: ResNet-50 (Transfer Learning)\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| **ROC-AUC** | {auc:.3f} |\n"
        f"| **Accuracy** | {acc:.3f} |\n"
        f"| **Recall** | {recall:.3f} |\n"
        f"| **Precision** | {precision:.3f} |\n"
        f"| **F1** | {f1:.3f} |\n"
        f"| **Threshold** | {THRESHOLD:.2f} |\n\n"
        f"**Baseline CNN AUC:** 0.611 &nbsp;→&nbsp; **ResNet-50 AUC:** {auc:.3f} "
        f"(+{auc-0.611:.3f})\n\n"
        f"Evaluated on {len(all_labels)} held-out test images. "
        f"Patient-level splits were used to prevent data leakage. "
        f"Threshold tuned on the validation set to maximise F1."
    )

    return fig1, fig2, summary

fig_curves, fig_cm, metrics_summary = make_performance_figures()

# ── Tab 1: Classifier ─────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return None
    img    = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    return {
        "Disease Present": round(prob, 4),
        "No Finding":      round(1 - prob, 4),
    }

classifier_tab = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a Chest X-ray"),
    outputs=gr.Label(label="Prediction", num_top_classes=2),
    title="NIH Chest X-ray Classifier",
    description=(
        f"Upload a frontal chest X-ray to classify it as **Disease Present** or **No Finding**.\n\n"
        f"Model: ResNet-50 fine-tuned on NIH Chest X-ray dataset · Decision threshold: {THRESHOLD:.2f}"
    ),
    flagging_mode="never",
)

# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with gr.Blocks() as performance_tab:
    gr.Markdown("## Model Performance")
    gr.Markdown(metrics_summary)
    gr.Plot(value=fig_curves, label="ROC and Precision-Recall Curves")
    gr.Plot(value=fig_cm,     label="Confusion Matrix")

# ── Combine into tabbed app ───────────────────────────────────────────────────
app = gr.TabbedInterface(
    [classifier_tab, performance_tab],
    tab_names=["Classifier", "Model Performance"],
    title="NIH Chest X-ray Classifier",
)

app.launch()
