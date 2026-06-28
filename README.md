# Early Detection of Thoracic Disease Using NIH Chest X-rays

**Live Demo:** https://huggingface.co/spaces/Doraemoons/nih-xray-classifier

End-to-end, cloud-based machine learning project focused on binary chest X-ray classification:

- `Disease Present`
- `No Finding`

This repository is organized for a 5-person Statistics and Data Science team building a resume-ready healthcare analytics project with strong ML and statistical rigor.

## Project Goals

- Access NIH Chest X-ray metadata and images programmatically from Google Cloud
- Build reproducible data engineering and preprocessing pipelines
- Train and compare baseline and transfer-learning image classifiers
- Evaluate performance under class imbalance with clinically relevant metrics
- Communicate model behavior, failures, and healthcare implications clearly

## Repository Structure

`docs/`
- Phase-by-phase plans, checklists, and implementation guides

`data/`
- Data dictionary, file conventions, and data handling documentation

`src/`
- Future Python package code (pipelines, modeling, evaluation)

`notebooks/`
- Exploratory and presentation notebooks (kept lightweight and reproducible)

`configs/`
- Future config files for experiments, data paths, and training

`reports/`
- Drafts and final artifacts (figures, report assets, presentation content)

`.github/`
- Collaboration templates (PR template, issue templates)

## Execution Plan

Follow milestones in `PROJECT_ROADMAP.md`.

Phase 1 starts with:
- Google Cloud setup
- BigQuery access
- Metadata pull into Python
- Reproducibility and validation checkpoints

Detailed Phase 1 checklist is in `docs/01_phase_1_cloud_setup.md`.

## Collaboration Workflow

- Branch and PR workflow is defined in `CONTRIBUTING.md`
- Team planning cadence and ownership matrix are in `TEAM_ROLES.md`
- Use issue templates for scoped tasks and progress visibility

## Current Status

- Phase 1 complete — BigQuery metadata ingestion
- Phase 2 complete — label engineering and patient-level train/val/test split
- Phase 3 complete — image pipeline and DataLoaders
- Phase 4 complete — baseline CNN trained and evaluated
- Phase 5 complete — transfer learning (ResNet-50), AUC 0.705
- Phase 6 complete — statistical evaluation, bootstrap CI, clinical interpretation
- Phase 7 complete — final report and portfolio packaging

## Results

| Model | AUC | Recall | Precision | F1 |
|-------|-----|--------|-----------|-----|
| Baseline CNN | 0.611 | — | — | — |
| ResNet-50 (Transfer) | **0.705** | **0.904** | 0.574 | 0.702 |

- ResNet-50 trained with two-phase transfer learning (frozen backbone → full fine-tune)
- Threshold tuned on validation set (0.11) to maximize F1
- High recall makes the model suitable as a screening triage tool
- Trained on 1,000 images per split — full dataset training expected to improve results significantly

See `notebooks/07_report.ipynb` for the complete project report.

---

## Getting Started (New Team Members)

Follow these steps to get the project running on your machine.

### 1. Clone the repo

```bash
git clone https://github.com/portablerug/x-ray-classification.git
cd x-ray-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the credentials file

You need the shared Google Cloud service account file to access BigQuery and Cloud Storage.
Request `nih-xray-ml-8be81f9160f0.json` from the team lead and place it in:

```
configs/nih-xray-ml-8be81f9160f0.json
```

Do not commit this file — it is already listed in `.gitignore`.
You do not need your own GCP account.

### 4. Download the image dataset

The full NIH Chest X-ray dataset (~45 GB) is available for free on Kaggle:

1. Create a free account at kaggle.com
2. Search for **NIH Chest X-rays** and download the dataset
3. Extract it to a folder on your machine (an external SSD is recommended due to size)
4. Note the path to the `archive/` folder — you will set this in the notebooks

### 5. Set your local image path

In `03_image_pipeline.ipynb` and `04_baseline_model.ipynb`, update `ARCHIVE_DIR` to match where you extracted the dataset:

```python
ARCHIVE_DIR = Path("/your/local/path/to/archive")
```

This is the only path that differs between teammates.

### 6. Run notebooks in order

| Notebook | What it does |
|----------|-------------|
| `01_data_loading.ipynb` | BigQuery connection and metadata pull |
| `02_label_engineering.ipynb` | Binary labels and train/val/test split |
| `03_image_pipeline.ipynb` | Image transforms and DataLoaders |
| `04_baseline_model.ipynb` | Baseline CNN training and evaluation |
| `05_transfer_learning.ipynb` | ResNet-50 transfer learning, threshold tuning |
| `05_transfer_learning.ipynb` | ResNet-50 transfer learning, threshold tuning |
| `06_evaluation.ipynb` | Statistical evaluation, bootstrap CI, clinical interpretation |
| `07_report.ipynb` | Final project report with live prediction demo |

Each notebook is self-contained and can be run independently.
