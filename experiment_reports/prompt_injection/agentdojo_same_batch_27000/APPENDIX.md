# AgentDojo role-perception appendix

Rows are true roles and columns predicted roles. Percentages are normalized within each true role; raw counts follow each percentage matrix. A zero-support row is unavailable, not zero accuracy. Ground truth follows message/channel metadata; attack text never changes its enclosing message role.

Every saved conversation is replayed whole using retrospective Harmony normalization, including metadata-derived tool identities and unscored tool schemas. This is not byte-identical serving-prompt recovery. AgentDojo discarded reasoning-channel text, so all five classes remain possible predictions but the Reasoning row has no support.

The expected denominators retain 95 clean and 921 attacked attempts per model. Available partial histories from context-limit errors are included, missing histories are unscorable, and histories above native 131,072-token context are excluded without truncation. Task-success denominators are unchanged.

| Subset | Model | Scored / assigned | Missing history | Above native context | Saved partial histories |
|---|---|---:|---:|---:|---:|
| Clean | Base | Pending | — | — | — |
| Attacked | Base | Pending | — | — | — |
| Clean | Same-batch KL (27,000) | Pending | — | — | — |
| Attacked | Same-batch KL (27,000) | Pending | — | — | — |

## Probe fitting

Two independent five-class multinomial linear probes are fitted from scratch on the original 250 C4/Dolma passages with their original train/test split, up to 1,024 passage tokens and every content token. Fits use the post-attention-layernorm/pre-MLP site at zero-based layer 12, C=0.005 and cuML max_iter=5,000. The role order, exact model/data hashes, token counts, stopping diagnostics and fit test accuracy are saved with each probe. Iteration count alone does not establish convergence; inspect job logs for cuML line-search warnings.

