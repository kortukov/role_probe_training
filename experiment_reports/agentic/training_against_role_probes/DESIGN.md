# First-run design for discussion

The user-confirmed objective is **improved role recoverability**. The model should
strengthen correct-role predictions, not hide roles or fool the probes.

## What one update means

1. Select a batch from the fixed AppWorld training corpus and run the current model.
2. Freeze the role probes and backpropagate correct-role loss through their inputs
   to a LoRA adapter. Include the KL anchor in this same optimizer update.
3. Freeze the model, recompute fitting activations, and update the probes against
   metadata labels. Old cached activations cannot be reused after a model update.
4. Log already computed batch metrics; periodically score fixed development data.

There is no trajectory generation inside this loop. Probe fitting and model
optimization are separate optimizers. Refit data remains fixed; representations
change. Do not split repeated attempts of one task between fitting and validation.

## Verified upstream choices

The pinned honesty example specifies 30 model updates; each samples eight honest
and eight deceptive prompts, plus 16 KL pairs. It uses AdamW, model learning rate
5e-5, weight decay 1e-4, LoRA rank 64 / alpha 128 / dropout zero, and KL weight 16.
These are update counts, not epochs. There is no direct honest/deceptive pairing
for conversational roles, so a 16-example role batch is an adaptation.

`probe_retrain_mode: retrain` warm-starts probe weights after every model update,
with 2000 probe optimization steps. It does not mean resetting weights each time.
`reinit` is the separate from-scratch regime. The probe learning rate is 1e-3,
batch size 32, weight decay 1e-4, cosine schedule and 50 warmup steps. These binary
hinge-trained probes differ from our five-class logistic probes.

The honesty KL anchor uses ordinary `allenai/Dolci-Instruct-SFT` training prompts
with answers generated once by the unmodified reasoning model. Its documented
construction uses 800 prompts, seed 42 and a 1536-token generation limit. This is
not the removed Dolci Tool-Use role-training dataset. For our experiment the base
answers must come from GPT-OSS-20B, not Qwen. The objective is KL(base || adapted),
with upstream completion-position masking, coefficient 16 and 16 pairs per update.
Use the upstream data selection and loss implementation where compatible; preserve
Harmony channels in our serialization. Exact original prompt identities are not
available in the committed upstream repository, so identical source/setup must not
be described as identical original sampled rows without verification.

## Recommended adaptations, not yet launched

- Start with layer 12's post-attention-layernorm output, keeping the five-class
  linear architecture. Use correct-label cross-entropy for model updates and
  warm-started probe refits. This is a deliberate multiclass adaptation of the
  paper's binary hinge objective; its numerical loss scale is different.
- Preserve all metadata-labeled content roles, excluding formatting tokens.
  Report both overall token accuracy and macro recall. Choose the role weighting
  explicitly before launching: equal tokens matches existing accuracy, whereas
  equal roles prevents reasoning/tool volume from dominating.
- Start from the paper's 30 model updates and 16 role examples per update, plus
  16 KL examples. Determine what constitutes a role example after tokenizing the
  corpus: a complete trajectory can be far longer than the paper's completion.
  Splitting into shorter sequences changes available context and must be reported.
  Do not silently truncate or discard most of each trajectory.
- Do not call this one epoch: 30 x 16 = 480 example presentations, potentially
  with reuse. Report unique trajectories, training sequences and labeled tokens
  seen; a full pass over all sequences is a different stopping rule.
- Match rank/alpha/learning rate/weight decay/KL weight first. GPT-OSS uses fused
  MoE expert parameters, unlike the upstream dense models. Validate adapter target
  coverage and a backward pass before promising the same target-module coverage.
- Log batch role loss/accuracy, KL, gradient norms and phase timings each update.
  Validate a small fixed development subset every five updates and at steps zero
  and 30. Keep raw confusion counts locally; put final matrices in APPENDIX.md.
  Fit independent diagnostic probes at the start/end, so co-adaptation is visible.
  Keep new trajectory generation out of frequent logging; evaluate task success
  separately at the start/end. Measure logging/validation overhead.

Before optimization, settle role weighting, layer selection, sequence unit/context
length and whether 30 updates or one complete corpus pass is the desired budget.
Collection does not depend on those choices.

## Measured sequence scale

Tokenizing the previous 90 training trajectories with our existing Harmony replay
produced 1,162,340 tokens: median 11,073, 90th percentile 19,773, maximum 33,532.
All 90 exceed 4096 tokens. These are planning measurements from the earlier corpus,
not counts for the new 300-attempt corpus; see `pilot_lengths.json`.

One concrete starting option is 4096-token segments, 16 segments per model update,
with gradient accumulation. Keep every segment in the corpus instead of truncating
each trajectory to its first segment. Later segments then lose earlier context;
this is an explicit approximation, requiring comparison with longer-context
validation. Thirty updates expose at most 1,966,080 token positions and are not
necessarily a full corpus pass. A full shuffled pass is an alternative budget,
whose update count should be calculated from the completed corpus.
