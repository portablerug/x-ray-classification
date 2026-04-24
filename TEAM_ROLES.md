# Team Roles and Coordination Plan (5 Members)

## Role Ownership

## 1) Data Engineer
- Owns cloud data access (BigQuery/GCS)
- Owns metadata extraction and data quality checks
- Maintains data dictionaries and schema notes

## 2) ML Engineer
- Owns model training pipeline architecture
- Owns baseline and transfer learning experiments
- Tracks reproducibility across runs

## 3) Statistician / Analyst
- Owns class imbalance strategy and metric framework
- Leads threshold analysis and model comparison rigor
- Ensures evaluation reflects healthcare risk priorities

## 4) Research / Domain Lead
- Owns clinical framing and literature grounding
- Translates model findings to healthcare implications
- Flags ethical and interpretation limitations

## 5) Visualization / Reporting
- Owns plotting standards and storytelling flow
- Builds report-ready figures and result narratives
- Leads final deck/report assembly and consistency

## Collaboration Rhythm

- Weekly planning meeting (scope and blockers)
- Mid-week async check-in (progress and risks)
- Weekly demo (what was completed and what changed)

## Git Workflow Responsibilities

- Every task linked to a GitHub issue
- Every feature/fix via pull request
- At least one reviewer from outside task owner role

## Definition of Done (Team Standard)

- Work has a clear owner and documented purpose
- Reproducibility notes included
- Tests/checks relevant to the task are completed
- Results are interpretable (not just "it runs")
- PR reviewed and merged with no unresolved blockers

## Handoff Protocol

When handing work to another role, include:

- What was completed
- What assumptions were made
- What files/artifacts changed
- What risks/open questions remain
- What exact next action is expected
