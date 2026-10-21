# Layer-12 role-probe transfer

## Motivation & RQ

Measure role-probe transfer to realistic agentic data and compare on-policy benchmark traces with off-policy Dolci trajectories.

## Evaluation setup

Metadata-based retrospective Harmony replay of the same trajectory schema. Capture zero-based layer 12 post-attention-layernorm output, not block output. Save per-token predictions/probabilities for every token; score content tokens only. Tool-schema context is excluded from accuracy. For traces over 16,384 tokens, reset causal context at explicit nonoverlapping window boundaries; keep every token. A single model forward per window uses the original eager attention to bound memory while retaining causal context. Dolci maximum replay length is 305,131 tokens, so one global forward is not feasible.

## Factual observations

Run in progress. No final performance claim yet. Data and predictions are excluded from Git.

## Message

Interpretation awaits completed runs. Detailed confusion matrices belong in [APPENDIX.md](APPENDIX.md#confusion-matrices).

See [RUNNING.md](RUNNING.md) for reproduction and standalone-viewer generation.
