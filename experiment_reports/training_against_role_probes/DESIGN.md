# First training run

The user-confirmed objective is **improved role recoverability**. The model should
strengthen correct-role predictions, not hide roles or fool the probes.

## What one update means

1. Select a batch from the fixed AppWorld training corpus and run the current model.
2. Freeze the role probes and backpropagate correct-role loss through their inputs
   to a LoRA adapter. Include the KL anchor in this same optimizer update.
3. Freeze the model, recompute activations on the standard role-confusion
   C4/Dolma passages, and refit probes against their assigned role labels. Old cached activations cannot be reused after a model update.
4. Log already computed batch metrics; periodically score fixed development data.

There is no trajectory generation inside this loop. Probe fitting and model
optimization are separate optimizers. Refit data remains fixed; representations
change. Probe fitting uses the existing 250 passages, five role renderings,
1,024-token limit and original train/test split, never AppWorld. AppWorld
model-update tasks and development tasks remain separate.

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
construction uses 800 prompts, seed 42 and a 1536-token generation limit. For this experiment the base answers come from GPT-OSS-20B. The objective is KL(base || adapted),
with upstream completion-position masking, coefficient 16 and 16 pairs per update.
Use the upstream data selection and loss implementation where compatible; preserve
Harmony channels in our serialization. Exact original prompt identities are not
available in the committed upstream repository, so identical source/setup must not
be described as identical original sampled rows without verification.

## Agreed choices and remaining design decisions

- Start with layer 12's post-attention-layernorm output, keeping the five-class
  linear architecture. Use correct-label cross-entropy for model updates and
  warm-started probe refits. This is a deliberate multiclass adaptation of the
  paper's binary hinge objective; its numerical loss scale is different.
- Preserve all metadata-labeled content roles, excluding formatting tokens.
  Report both overall token accuracy and macro recall. The first run weights content tokens equally, matching the throughput
  calibration. Roles are not artificially rebalanced.
- The training target is **2,700 trajectories**, 30 attempts for each of 90
  tasks. Existing 300 trajectories are debug data only. At an effective batch of
  16 complete trajectories, one full training pass takes 169 model updates;
  30 updates cover only about 17.8% of the corpus. The first run is one epoch; this is an explicit adaptation of the paper’s
  30-update budget.
- Preserve complete trajectories, using two trajectories per GPU microbatch and
  accumulating gradients across trajectories if necessary. Padding is supported
  in principle, but avoid it where it adds unnecessary computation. If full
  trajectories prove infeasible, discuss start-anchored prefixes preserving several
  turns; do not replace them with arbitrary middle-of-trajectory segments.
- Match rank/alpha/learning rate/weight decay/KL weight first. GPT-OSS uses fused
  MoE expert parameters, unlike the upstream dense models. Validate adapter target
  coverage and a backward pass before promising the same target-module coverage.
- Log batch role loss/accuracy, KL, gradient norms and phase timings each update.
  Validate a small fixed development subset every five updates and at steps zero
  and 30. Keep raw confusion counts locally; put final matrices in APPENDIX.md.
  Fit independent diagnostic probes at the start/end, so co-adaptation is visible.
  Keep new trajectory generation out of frequent logging; evaluate task success
  separately at the start/end. Measure logging/validation overhead.

The first run uses layer 12, token-weighted cross-entropy, effective batch16,
microbatch2 per GPU and one epoch. Standard-data refits use 2000 SGD steps,
32 passage-role examples per step with every content token retained; this
multiclass, token-level objective adapts the upstream binary sequence objective.

## Debug-corpus measurements and batching

The completed debug corpus has 300 trajectories and 4,098,694 tokens. Median length
is 12,466, the 95th percentile 24,041 and maximum 33,241; median message count is 44.
At 16 trajectories/update, this debug corpus takes 19 updates per epoch. These
counts do not describe the planned 2,700-trajectory training corpus.

Random batches of 16 allocate 7,638,512 padded token positions for this corpus:
46.3% padding, or 1.86× unpadded token work. Sorting into similar-length groups
reduces padding to 5.1%. These are token-allocation estimates, not measured runtime.
A microbatch of one full trajectory needs no padding. Weight accumulated gradients
by the chosen loss denominator, so microbatching preserves the effective-batch
objective rather than accidentally averaging trajectories equally.

All debug trajectories fit the model's configured 131,072-position limit; one
exceeds the previous inference server's 32,768 limit. Training memory is a separate
constraint and must be tested with efficient attention and gradient checkpointing.

Extrapolating the debug mean to 2,700 trajectories gives approximately 36.9 million
tokens for model updates. Probe refitting has its own fixed standard-data cost;
it does not re-extract the AppWorld corpus.

## Independent probe check in plain language

The main check is not applying the old base-model probe to the final model.
Instead, freeze the base model and fit a new probe; freeze the final model and
fit another new probe with the same standard C4/Dolma training data and budget. Evaluate each probe on its
own model's activations for the same held-out tasks. Does a probe learning from
scratch recover roles more accurately on the trained model?

Yes, the final check is essentially one extra probe fit at the end. The difference
is resetting its weights rather than continuing the probe that supplied the
training gradients. It is an optional diagnostic, not another model-training loop.
An old probe transferred directly to the final model instead tests whether its
original decision boundary still works, which is a different question.

Authentication is verified on the user's W&B server. Data collection is active.
The complete training loop is implemented and awaiting its GPU pilot, including
KL, standard-data refitting, coordinated microbatch fallback and checkpoint/resume.
The communication file will announce readiness only after tests pass and the
first AppWorld training run starts.
