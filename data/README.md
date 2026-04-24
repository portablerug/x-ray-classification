# Data Directory Guide

Use this directory for data-related documentation and small tracked metadata artifacts.

## Intended Subdirectories

- `raw/` (ignored): unmodified source data snapshots
- `interim/` (ignored): temporary transformation outputs
- `processed/` (ignored): modeling-ready datasets

## Rules

- Do not commit large image datasets to Git
- Document every dataset version used in experiments
- Keep a clear data dictionary for major columns/fields
