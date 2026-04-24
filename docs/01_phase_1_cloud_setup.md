# Phase 1 Guide: Google Cloud + BigQuery Metadata Access

This phase establishes secure, reproducible cloud data access for NIH Chest X-ray metadata.

## Objectives

- Set up Google Cloud project permissions for team access
- Validate BigQuery connectivity from Python environment
- Define metadata query scope and output schema
- Build traceable mapping from image identifiers to GCS paths

## Scope Checklist

- [ ] Confirm Google Cloud project, billing, and IAM roles
- [ ] Confirm dataset/table names and access permissions
- [ ] Define required metadata fields (IDs, labels, split-relevant fields)
- [ ] Define binary label rule (`Disease Present` vs `No Finding`)
- [ ] Define output location and naming convention
- [ ] Define quality checks (null counts, duplicate IDs, class distribution)

## Reproducibility Requirements

- Record environment assumptions
- Record query version and timestamp
- Save row counts and basic quality stats
- Keep data access steps scriptable and deterministic

## Risks to Watch Early

- Inconsistent label strings and multi-label formatting
- Missing image links for some metadata entries
- Access permission mismatch across team members
- Silent schema changes in cloud-hosted sources

## Phase 1 Exit Criteria

- Team can programmatically pull metadata and verify schema
- Binary target definition is documented and agreed
- GCS path construction rule is validated for sample records
- Quality checks are complete and reviewed
