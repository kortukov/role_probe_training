# Missing-layer probe fitting

## Motivation & RQ

Fit baseline-compatible probes for layers 4, 8 and 20 for the requested agentic comparison.

## Evaluation setup

Exact original documents/splits, 250 passages, full content-token scale, C=0.005; only selected layers differ. Layer 12/16 weights remain unchanged. Job 1460163.

## Factual observations

Training completed (job 1460163). Held-out synthetic token accuracy: layer 4 46.70%, layer 8 77.49%, layer 20 77.11%. All three fits emitted cuML L-BFGS line-search warnings and stopped at the last valid step; these are usable saved fits, not certified optimizer convergence. See `metrics/synthetic.json` and matching probe metadata.

## Message

Synthetic role-header performance varies substantially by layer. Transfer interpretation awaits [agentic evaluation](../../agentic/probe_layer_comparison/README.md).

See [RUNNING.md](RUNNING.md).
