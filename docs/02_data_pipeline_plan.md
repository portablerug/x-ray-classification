# Phase 2 Guide: Data Pipeline and Label Engineering

## Objectives

- Clean metadata into analysis-ready format
- Engineer binary target with explicit rules
- Preserve provenance and reproducibility

## Decisions to Finalize

- Label parsing and normalization strategy
- Missing/ambiguous label policy
- Duplicate image/patient handling policy
- Split strategy to avoid leakage

## Quality Checks

- [ ] Nulls by column
- [ ] Duplicate IDs and potential duplicates by patient
- [ ] Class balance by split
- [ ] Label distribution before and after cleaning

## Artifacts Expected

- Data dictionary update
- Label engineering assumptions doc
- Validation summary for cleaned dataset
