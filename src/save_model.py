"""
Trains ResNet-50 and saves model weights + predictions to models/.
Designed to run from the command line without needing a notebook kernel.
Uses reduced settings (300 images/split, 2+5 epochs) for CPU speed.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from google.cloud import bigquery
from google.oauth2 import service_account
from PIL import Image

BASE_DIR      = Path(__file__).resolve().parent.parent
CRED_PATH     = BASE_DIR / "configs" / "nih-xray-ml-8be81f9160f0.json"
IMAGE_DIR     = Path("/Volumes/T7 Shield/x-ray-datasets/archive/images_006/images")
MODELS_DIR    = BASE_DIR / "models"
MAX_PER_SPLIT = 300
N_EPOCHS_A    = 2
N_EPOCHS_B    = 5

MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Connecting to BigQuery...")
credentials = service_account.Credentials.from_service_account_file(str(CRED_PATH))
bq_client   = bigquery.Client(project="nih-xray-ml", credentials=credentials)

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

local_files    = set(p.name for p in IMAGE_DIR.glob("*.png"))
image_path_map = {p.name: p for p in IMAGE_DIR.glob("*.png")}
df_available   = df[df["Image Index"].isin(local_files)].reset_index(drop=True)

parts = []
for split in ["train", "val", "test"]:
    subset = df_available[df_available["split"] == split]
    parts.append(subset.sample(n=min(MAX_PER_SPLIT, len(subset)), random_state=42))
df_available = pd.concat(parts).reset_index(drop=True)

print(f"Images loaded: {len(df_available)} ({MAX_PER_SPLIT}/split)")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

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

train_ds = XRayDataset(df_available[df_available["split"] == "train"], image_path_map, train_transform)
val_ds   = XRayDataset(df_available[df_available["split"] == "val"],   image_path_map, val_test_transform)
test_ds  = XRayDataset(df_available[df_available["split"] == "test"],  image_path_map, val_test_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()

# Phase A
print(f"\nPhase A: {N_EPOCHS_A} epochs (frozen backbone)...")
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer_A = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
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
    print(f"  Epoch {epoch}/{N_EPOCHS_A} | Train Loss: {train_loss/len(train_loader):.4f}")

# Phase B
print(f"\nPhase B: {N_EPOCHS_B} epochs (full fine-tune)...")
for param in model.parameters():
    param.requires_grad = True

optimizer_B = torch.optim.Adam(model.parameters(), lr=1e-5)
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
    print(f"  Epoch {epoch}/{N_EPOCHS_B} | Train Loss: {train_loss/len(train_loader):.4f}")

# Threshold tuning on val
print("\nTuning threshold on validation set...")
model.eval()
val_probs, val_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy()
        val_probs.extend(probs)
        val_labels.extend(labels.numpy())

val_probs  = np.array(val_probs)
val_labels = np.array(val_labels)
thresholds = np.arange(0.01, 1.0, 0.01)
f1s        = [f1_score(val_labels, (val_probs >= t).astype(int), zero_division=0) for t in thresholds]
best_threshold = thresholds[np.argmax(f1s)]
print(f"Best threshold: {best_threshold:.2f}")

# Test set predictions
print("\nRunning test set predictions...")
all_probs, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
auc        = roc_auc_score(all_labels, all_probs)
print(f"Test AUC: {auc:.3f}")

# Save everything
torch.save(model.state_dict(), MODELS_DIR / "resnet50_phase5.pt")
pd.DataFrame({"true_label": all_labels, "prob_disease": all_probs}).to_csv(
    MODELS_DIR / "predictions_resnet50.csv", index=False)
(MODELS_DIR / "best_threshold.txt").write_text(str(best_threshold))

print(f"\nSaved -> models/resnet50_phase5.pt")
print(f"Saved -> models/predictions_resnet50.csv")
print(f"Saved -> models/best_threshold.txt")
print("\nDone. You can now run the Gradio app.")
