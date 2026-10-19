# PR title

Document W6 popularity and long-tail recommendation experiments

## Summary

- Preserve the original W6 baseline metrics and Top-10 artifacts.
- Add rerunnable audit and sweep commands with output-directory options so new runs can use separate folders.
- Document four primary baselines, observed-only MF convergence status, and long-tail relevance results.
- Add an Overleaf-ready Method draft, Progress Report summary, and AI/contribution/review record templates.

## Validation notes

- All 11,850 evaluated users receive 10 unique recommendations with seen items filtered.
- Validation was used for parameter selection; no independent test split was supplied.
- Observed-only MF at 1,000 iterations is not converged and is not promoted to the main baseline.
- Human review and member contribution evidence remain pending.
