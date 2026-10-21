# Role-probe layer comparison

## Motivation & RQ

Measure role-probe transfer to realistic agentic data and compare on-policy benchmark traces with off-policy Dolci trajectories.

## Evaluation setup

Layers 4, 8, 12, 16, 20 on exactly the same trajectories. Reuse baseline layers 12/16; train missing layers 4/8/20 with exactly the original 250 documents, all content tokens, same splits and logistic-regression settings. Capture all five sites in one replay. This is a separate analysis of the shared predictions, not a new sample.

## Factual observations

Run in progress. No final performance claim yet. Data and predictions are excluded from Git.

## Message

Interpretation awaits completed runs. Detailed confusion matrices belong in [APPENDIX.md](APPENDIX.md#confusion-matrices).

See [RUNNING.md](RUNNING.md) for reproduction and standalone-viewer generation.
