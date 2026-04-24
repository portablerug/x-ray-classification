# Project Roadmap and Milestones

This roadmap breaks the project into practical phases with clear outputs, ownership, and review checkpoints.

## Phase 0 - Team and Repo Setup

### Objectives
- Establish repository standards and collaboration process
- Align role ownership and communication cadence

### Outputs
- Repository scaffold
- Role matrix and milestone plan
- PR and issue templates

### Exit Criteria
- Every team member has role ownership and at least one Phase 1 task

## Phase 1 - Cloud Access and Metadata Ingestion

### Objectives
- Configure Google Cloud project access
- Query NIH metadata from BigQuery
- Build reproducible metadata ingestion process

### Outputs
- Validated BigQuery query plan
- Metadata pull specification (fields, filters, row counts)
- Reproducibility checklist

### Statistical Notes
- Define positive class (`Disease Present`) and negative class (`No Finding`)
- Record class distribution baseline before modeling

### Exit Criteria
- Team can programmatically retrieve metadata and map image IDs to GCS paths

## Phase 2 - Data Cleaning and Label Engineering

### Objectives
- Standardize labels and create binary target
- Handle uncertain/missing/noisy label scenarios
- Finalize train/validation/test split policy

### Outputs
- Label mapping policy doc
- Data quality report
- Split strategy with leakage checks

### Exit Criteria
- Clean analysis-ready metadata table with binary outcome

## Phase 3 - Image Pipeline and Feature Preparation

### Objectives
- Define image retrieval strategy (download vs stream)
- Standardize transformations (resize/normalize)
- Build memory-safe batch handling strategy

### Outputs
- Image preprocessing specification
- Data loader strategy and throughput benchmarks
- Pipeline QA checklist

### Exit Criteria
- Reproducible image pipeline feeds modeling stage without memory failures

## Phase 4 - Baseline Modeling

### Objectives
- Build baseline classifier to establish benchmark
- Produce initial error profile and metric baseline

### Outputs
- Baseline model experiment log
- Baseline metrics (precision, recall, F1, ROC-AUC, confusion matrix)

### Exit Criteria
- Baseline results reviewed and approved as benchmark

## Phase 5 - Transfer Learning and Model Comparison

### Objectives
- Train transfer-learning model(s) for improved performance
- Compare against baseline under class imbalance constraints

### Outputs
- Model comparison report
- Training decision log and hyperparameter notes

### Exit Criteria
- Best candidate model selected with transparent trade-off analysis

## Phase 6 - Statistical Evaluation and Interpretation

### Objectives
- Deep evaluation under class imbalance
- Analyze false negatives, threshold trade-offs, and clinical implications

### Outputs
- Evaluation narrative with metric trade-off discussion
- Error analysis by subgroup/label pattern if feasible

### Exit Criteria
- Results are interpretable and decision-ready for stakeholder communication

## Phase 7 - Reporting, Portfolio Packaging, and Demo

### Objectives
- Finalize report, visuals, and project story
- Produce resume bullets and interview-ready talking points

### Outputs
- Final report and presentation deck
- GitHub README polishing and project highlights
- Optional lightweight dashboard/demo

### Exit Criteria
- Repository is portfolio-ready and reproducible

## Weekly Milestone Cadence (Suggested)

- Week 1: Phase 0-1
- Week 2: Phase 2
- Week 3: Phase 3-4
- Week 4: Phase 5
- Week 5: Phase 6
- Week 6: Phase 7
