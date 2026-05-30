# Early Detection of Thoracic Disease Using NIH Chest X-rays

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

## Team Roles

- Data Engineer
- ML Engineer
- Statistician / Analyst
- Research / Domain Lead
- Visualization / Reporting

Role ownership and responsibilities are documented in `TEAM_ROLES.md`.

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
