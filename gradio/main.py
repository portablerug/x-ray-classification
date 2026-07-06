"""
NIH Chest X-ray Classifier — Single-file pipeline
--------------------------------------------------
First run  : connects to BigQuery, loads images from local SSD,
             trains ResNet-50, saves weights + predictions.
After that : detects saved weights, skips training, launches app immediately.

To force retrain: delete models/resnet50_phase5.pt and run again.

Usage:
    python3 gradio/main.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, precision_recall_curve, average_precision_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# Set to None to train on all available images (recommended with GPU on Colab)
# Set to e.g. 300 for a quick local CPU test run
MAX_PER_SPLIT = None

IMAGE_DIR  = Path("/Volumes/T7 Shield/x-ray-datasets/archive/images_006/images")
N_EPOCHS_A = 3     # frozen backbone warm-up
N_EPOCHS_B = 15    # full fine-tune
LR_A       = 1e-3
LR_B       = 1e-5
BATCH_SIZE = 32

CRED_FILENAME = "nih-xray-ml-8be81f9160f0.json"
PROJECT_ID    = "nih-xray-ml"

# ── Paths ─────────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
if os.environ.get("SPACE_ID"):
    MODELS_DIR = _here          # Hugging Face: files sit next to app.py
else:
    MODELS_DIR = _here.parent.parent / "models"   # local: project_root/models/

MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = MODELS_DIR / "resnet50_phase5.pt"
THRESH_PATH = MODELS_DIR / "best_threshold.txt"
PREDS_PATH  = MODELS_DIR / "predictions_resnet50.csv"

# ── Shared transform ──────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, df, image_path_map, transform=None):
        self.df             = df.reset_index(drop=True)
        self.image_path_map = image_path_map
        self.transform      = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.image_path_map[row["Image Index"]]).convert("RGB")
        label = int(row["disease_present"])
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """Generates Grad-CAM heatmaps from the last ResNet-50 convolutional block."""

    def __init__(self, model, target_layer):
        self.model       = model
        self._activations = None
        self._gradients  = None

        self._fwd_hook = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_activations", o)
        )
        self._bwd_hook = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_gradients", go[0].detach())
        )

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        prob   = torch.sigmoid(output[0, 0]).item()

        output[0, 0].backward()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self._activations.detach()).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.squeeze().cpu().numpy(), prob

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_gradcam(original_img, cam, alpha=0.45):
    """Blend a Grad-CAM heatmap over the original PIL image."""
    cam_uint8   = (cam * 255).astype(np.uint8)
    cam_resized = Image.fromarray(cam_uint8).resize(original_img.size, Image.BILINEAR)
    cam_arr     = np.array(cam_resized) / 255.0

    colormap = (plt.cm.jet(cam_arr)[:, :, :3] * 255).astype(np.uint8)
    heatmap  = Image.fromarray(colormap).convert("RGB")

    return Image.blend(original_img.convert("RGB"), heatmap, alpha=alpha)

# ── Training pipeline ─────────────────────────────────────────────────────────
def run_training():
    print("No saved weights found — starting training pipeline.")

    from google.cloud import bigquery
    from google.oauth2 import service_account

    search_roots = [Path.cwd(), *Path.cwd().parents[:4]]
    cred_path = next(
        (r / "configs" / CRED_FILENAME for r in search_roots
         if (r / "configs" / CRED_FILENAME).exists()), None
    )
    if cred_path is None:
        raise FileNotFoundError(
            f"Credentials file '{CRED_FILENAME}' not found in any configs/ folder."
        )

    print("Connecting to BigQuery...")
    credentials = service_account.Credentials.from_service_account_file(str(cred_path))
    bq_client   = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    df = bq_client.query("""
        SELECT `Image Index`, `Finding Labels`
        FROM `nih-xray-ml.nih_xray.metadata`
    """).to_dataframe(create_bqstorage_client=False)

    df["disease_present"] = (df["Finding Labels"] != "No Finding").astype(int)
    df["patient_id"]      = df["Image Index"].str.split("_").str[0]

    patients = df["patient_id"].unique()
    train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
    val_p,   test_p = train_test_split(temp_p,   test_size=0.50, random_state=42)

    df["split"] = "train"
    df.loc[df["patient_id"].isin(val_p),  "split"] = "val"
    df.loc[df["patient_id"].isin(test_p), "split"] = "test"

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}\nUpdate IMAGE_DIR in the configuration section.")

    image_path_map = {p.name: p for p in IMAGE_DIR.glob("*.png")}
    df_available   = df[df["Image Index"].isin(image_path_map)].reset_index(drop=True)

    if MAX_PER_SPLIT is not None:
        parts = []
        for split in ["train", "val", "test"]:
            subset = df_available[df_available["split"] == split]
            parts.append(subset.sample(n=min(MAX_PER_SPLIT, len(subset)), random_state=42))
        df_available = pd.concat(parts).reset_index(drop=True)

    print(f"Images available: {len(df_available):,}")
    for split in ["train", "val", "test"]:
        s = df_available[df_available["split"] == split]
        print(f"  {split:6}: {len(s):,} images | {s['disease_present'].mean():.1%} disease")

    train_ds = XRayDataset(df_available[df_available["split"] == "train"], image_path_map, train_transform)
    val_ds   = XRayDataset(df_available[df_available["split"] == "val"],   image_path_map, val_test_transform)
    test_ds  = XRayDataset(df_available[df_available["split"] == "test"],  image_path_map, val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model     = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc  = nn.Linear(model.fc.in_features, 1)
    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Phase A — frozen backbone
    print(f"\nPhase A: {N_EPOCHS_A} epochs (frozen backbone, lr={LR_A})")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer_A = torch.optim.Adam(model.fc.parameters(), lr=LR_A)
    for epoch in range(1, N_EPOCHS_A + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            optimizer_A.zero_grad()
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            optimizer_A.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                out    = model(images).squeeze(1)
                val_loss += criterion(out, labels).item()
                preds    = (torch.sigmoid(out) >= 0.5).long()
                correct  += (preds == labels.long()).sum().item()
                total    += labels.size(0)
        print(f"  Epoch {epoch}/{N_EPOCHS_A} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {correct/total:.3f}")

    # Phase B — full fine-tune
    print(f"\nPhase B: {N_EPOCHS_B} epochs (full fine-tune, lr={LR_B})")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_B = torch.optim.Adam(model.parameters(), lr=LR_B)
    for epoch in range(1, N_EPOCHS_B + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            optimizer_B.zero_grad()
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            optimizer_B.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                out    = model(images).squeeze(1)
                val_loss += criterion(out, labels).item()
                preds    = (torch.sigmoid(out) >= 0.5).long()
                correct  += (preds == labels.long()).sum().item()
                total    += labels.size(0)
        print(f"  Epoch {epoch:2d}/{N_EPOCHS_B} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {correct/total:.3f}")

    # Threshold tuning on val
    print("\nTuning threshold on validation set...")
    model.eval()
    val_probs, val_labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy()
            val_probs.extend(probs)
            val_labels_list.extend(labels.numpy())

    val_probs_arr  = np.array(val_probs)
    val_labels_arr = np.array(val_labels_list)
    thresholds     = np.arange(0.01, 1.0, 0.01)
    f1s            = [f1_score(val_labels_arr, (val_probs_arr >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1s)]
    print(f"Best threshold: {best_threshold:.2f}")

    # Test set predictions
    print("Running test set evaluation...")
    all_probs_list, all_labels_list = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy()
            all_probs_list.extend(probs)
            all_labels_list.extend(labels.numpy())

    all_probs_arr  = np.array(all_probs_list)
    all_labels_arr = np.array(all_labels_list)
    auc            = roc_auc_score(all_labels_arr, all_probs_arr)
    print(f"Test AUC: {auc:.3f}")

    # Save everything
    torch.save(model.state_dict(), MODEL_PATH)
    pd.DataFrame({"true_label": all_labels_arr, "prob_disease": all_probs_arr}).to_csv(PREDS_PATH, index=False)
    THRESH_PATH.write_text(str(best_threshold))

    print(f"\nSaved -> {MODEL_PATH}")
    print(f"Saved -> {PREDS_PATH}")
    print(f"Saved -> {THRESH_PATH}")
    print("\nTraining complete. Launching app...\n")

# ── Run training if no weights exist ─────────────────────────────────────────
if not MODEL_PATH.exists():
    run_training()
else:
    print(f"Weights found at {MODEL_PATH} — skipping training.")

# ── Load model for inference ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

THRESHOLD = float(THRESH_PATH.read_text().strip())

# Attach Grad-CAM to the last ResNet-50 convolutional block
gradcam = GradCAM(model, model.layer4[-1])

# ── Load test predictions for performance tab ─────────────────────────────────
df_preds   = pd.read_csv(PREDS_PATH)
all_labels = df_preds["true_label"].values
all_probs  = df_preds["prob_disease"].values
all_preds  = (all_probs >= THRESHOLD).astype(int)

# ── Pre-generate performance figures ──────────────────────────────────────────
def make_performance_figures():
    auc       = roc_auc_score(all_labels, all_probs)
    acc       = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)

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

    fig2, ax3 = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm, display_labels=["No Finding", "Disease"]).plot(
        ax=ax3, colorbar=False, cmap="Blues")
    ax3.set_title(f"Confusion Matrix  (threshold={THRESHOLD:.2f})")
    fig2.tight_layout()

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
        f"(+{auc - 0.611:.3f})\n\n"
        f"Evaluated on {len(all_labels)} held-out test images. "
        f"Patient-level splits prevent data leakage. "
        f"Threshold tuned on validation set to maximise F1."
    )

    return fig1, fig2, summary

fig_curves, fig_cm, metrics_summary = make_performance_figures()

# ── Predict + Grad-CAM ────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return None, None

    original_img = Image.fromarray(image).convert("RGB")
    tensor       = val_test_transform(original_img).unsqueeze(0).to(device)

    with torch.enable_grad():
        cam_map, prob = gradcam.generate(tensor)

    label_dict = {
        "Disease Present": round(prob, 4),
        "No Finding":      round(1 - prob, 4),
    }

    overlay = overlay_gradcam(original_img, cam_map)

    return label_dict, overlay

# ── Tab 1: Classifier ─────────────────────────────────────────────────────────
with gr.Blocks() as classifier_tab:
    gr.Markdown(
        f"## NIH Chest X-ray Classifier\n\n"
        f"Upload a frontal chest X-ray. The model returns a prediction and a "
        f"**Grad-CAM heatmap** showing which regions of the image drove the decision — "
        f"red/yellow = high attention, blue = low attention.\n\n"
        f"*Model: ResNet-50 · Threshold: {THRESHOLD:.2f} · Not for clinical use.*"
    )
    with gr.Row():
        img_input   = gr.Image(label="Upload Chest X-ray")
        label_out   = gr.Label(label="Prediction", num_top_classes=2)
    gradcam_out = gr.Image(label="Grad-CAM Heatmap — regions the model focused on")
    submit_btn  = gr.Button("Classify", variant="primary")
    submit_btn.click(fn=predict, inputs=img_input, outputs=[label_out, gradcam_out])

# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with gr.Blocks() as performance_tab:
    gr.Markdown("## Model Performance")
    gr.Markdown(metrics_summary)
    gr.Plot(value=fig_curves, label="ROC and Precision-Recall Curves")
    gr.Plot(value=fig_cm,     label="Confusion Matrix")

# ── Launch ────────────────────────────────────────────────────────────────────
app = gr.TabbedInterface(
    [classifier_tab, performance_tab],
    tab_names=["Classifier", "Model Performance"],
    title="NIH Chest X-ray Classifier",
)

app.launch()
