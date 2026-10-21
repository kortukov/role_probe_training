# Previous role-confusion work

Inspected on 2026-10-21 at the user's request:
`/home/fe/kortukov/workdir/prompt_injection_interp/head_analysis/role_confusion/`.
This is a reference source, not a runtime dependency of the new pipeline.

Useful components:

- `paper_reproduction/`: neutral C4/Dolma training, saved layer-12 weights for
  eight role subsets, and conversation-transfer summary tables.
- `shared/paper_protocol.py`: confirms the upstream activation site is the
  **output of post_attention_layernorm**, immediately before the MLP. A decoder
  block's output is a different site. The upstream `run_and_export_states`
  defaults to `all_pre_mlp_hidden_states`.
- `agent_dojo_clean/`: existing multi-layer SUCAT results show that strong
  controlled role discrimination does not guarantee accurate tool labeling on
  natural agent traces. Keep that limitation in mind for future tau-bench work.

Reused now: only the five-role (`sucat`) coefficients/intercepts at layer 12,
extracted from the 320-KB multi-probe NPZ into a smaller portable NPZ, plus
provenance. They live in
`experiment_reports/role_probes/gptoss20b_baseline/reference/` and provide an
independent historical comparison on our current evaluation activations.
They are not the new trained probes. No historical attack corpora, defense
adapters, activation caches, or trajectory viewers were imported.

The retained historical output directory has summary CSVs, weights, role-space
metadata and execution metadata. The referenced raw conversations and controlled
source texts are absent there; do not claim to have reused them. Historical model
revision matches ours. Historical extraction used native MXFP4 / FlashAttention,
whereas this run initially uses dequantized BF16 / eager attention. Comparisons
must identify this execution difference.

Read code as well as reports: the prior README describes a document-level split,
but `paper_data_analyze.py:fit_probes` splits role-wrapped prompt IDs. Our new split
is explicitly document-grouped; summaries are not sufficient provenance for an
exact protocol claim.
