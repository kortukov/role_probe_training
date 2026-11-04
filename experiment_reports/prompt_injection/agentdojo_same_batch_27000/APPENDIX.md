# AgentDojo role-perception appendix

Rows are true roles and columns predicted roles. Percentages are normalized within each true role; raw counts follow each percentage matrix. A zero-support row is unavailable, not zero accuracy. Ground truth follows message/channel metadata; attack text never changes its enclosing message role.

Every saved conversation is replayed whole using retrospective Harmony normalization, including metadata-derived tool identities and unscored tool schemas. This is not byte-identical serving-prompt recovery. AgentDojo discarded reasoning-channel text, so all five classes remain possible predictions but the Reasoning row has no support.

The expected denominators retain 95 clean and 921 attacked attempts per model. Available partial histories from context-limit errors are included, missing histories are unscorable, and histories above native 131,072-token context are excluded without truncation. Task-success denominators are unchanged.

| Subset | Model | Scored / assigned | Missing history | Above native context | Saved partial histories |
|---|---|---:|---:|---:|---:|
| Clean | Base | 95/95 | 0 | 0 | 0 |
| Attacked | Base | 921/921 | 0 | 0 | 5 |
| Clean | Same-batch KL (27,000) | 95/95 | 0 | 0 | 0 |
| Attacked | Same-batch KL (27,000) | 918/921 | 3 | 0 | 5 |

## Probe fitting

Two independent five-class multinomial linear probes are fitted from scratch on the original 250 C4/Dolma passages with their original train/test split, up to 1,024 passage tokens and every content token. Fits use the post-attention-layernorm/pre-MLP site at zero-based layer 12, C=0.005 and cuML max_iter=5,000. The role order, exact model/data hashes, token counts, stopping diagnostics and fit test accuracy are saved with each probe. Iteration count alone does not establish convergence; inspect job logs for cuML line-search warnings.

### Observed fit diagnostics

The base fit stopped at 390 iterations with `L-BFGS line search failed (code 1)`;
this does not establish convergence despite stopping below the 5,000-iteration
ceiling. The trained fit stopped at 560 iterations with no cuML warning found in
its log. Both use 642,530 training content tokens and 56,510 held-out content
tokens. The Python warning arrays do not capture the base's C++ log warning.
[Audited log diagnostics](role_fit_audit.json) retain the exact message. No probe
weights or reported predictions were changed after this audit.

## Base: clean

Overall token accuracy: **21.6%**. Scored content tokens: 105,113.

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 20.31% | 13.97% | 26.19% | 0.00% | 39.53% |
| User | 0.37% | 83.43% | 15.10% | 0.00% | 1.10% |
| Reasoning | — | — | — | — | — |
| Assistant | 1.75% | 4.55% | 24.80% | 58.38% | 10.53% |
| Tool | 16.75% | 32.24% | 34.59% | 6.08% | 10.34% |

Raw token counts:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 1968 | 1354 | 2538 | 0 | 3830 |
| User | 17 | 3785 | 685 | 0 | 50 |
| Reasoning | 0 | 0 | 0 | 0 | 0 |
| Assistant | 276 | 716 | 3906 | 9196 | 1659 |
| Tool | 12585 | 24220 | 25986 | 4571 | 7771 |

[Metrics and provenance](role_perception/base/clean.json) · [Fresh probe metadata](role_perception/base/post_training/probes/layer_12.json)

## Base: attacked

Overall token accuracy: **17.7%**. Scored content tokens: 1,690,124.

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 21.57% | 12.66% | 26.61% | 0.00% | 39.16% |
| User | 0.42% | 82.67% | 15.93% | 0.00% | 0.98% |
| Reasoning | — | — | — | — | — |
| Assistant | 3.17% | 4.83% | 30.62% | 51.00% | 10.39% |
| Tool | 18.42% | 31.97% | 31.14% | 7.47% | 11.00% |

Raw token counts:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 20262 | 11896 | 24994 | 0 | 36790 |
| User | 176 | 35034 | 6752 | 0 | 417 |
| Reasoning | 0 | 0 | 0 | 0 | 0 |
| Assistant | 5794 | 8835 | 56007 | 93290 | 18998 |
| Tool | 252495 | 438273 | 426894 | 102441 | 150776 |

[Metrics and provenance](role_perception/base/attacked.json) · [Fresh probe metadata](role_perception/base/post_training/probes/layer_12.json)

## Same-batch KL (27,000): clean

Overall token accuracy: **92.3%**. Scored content tokens: 117,458.

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 92.44% | 5.01% | 0.59% | 0.00% | 1.97% |
| User | 4.89% | 94.97% | 0.04% | 0.00% | 0.09% |
| Reasoning | — | — | — | — | — |
| Assistant | 0.13% | 0.04% | 0.17% | 57.98% | 41.69% |
| Tool | 1.21% | 0.67% | 0.14% | 0.00% | 97.98% |

Raw token counts:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 8957 | 485 | 57 | 0 | 191 |
| User | 222 | 4309 | 2 | 0 | 4 |
| Reasoning | 0 | 0 | 0 | 0 | 0 |
| Assistant | 19 | 6 | 25 | 8658 | 6226 |
| Tool | 1071 | 589 | 120 | 0 | 86517 |

[Metrics and provenance](role_perception/trained/clean.json) · [Fresh probe metadata](role_perception/trained/post_training/probes/layer_12.json)

## Same-batch KL (27,000): attacked

Overall token accuracy: **92.5%**. Scored content tokens: 1,705,195.

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 91.33% | 6.00% | 0.42% | 0.00% | 2.26% |
| User | 4.46% | 95.41% | 0.07% | 0.00% | 0.07% |
| Reasoning | — | — | — | — | — |
| Assistant | 0.17% | 0.29% | 0.38% | 48.24% | 50.91% |
| Tool | 1.22% | 0.55% | 0.14% | 0.00% | 98.09% |

Raw token counts:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 85514 | 5617 | 389 | 0 | 2116 |
| User | 1885 | 40367 | 28 | 0 | 29 |
| Reasoning | 0 | 0 | 0 | 0 | 0 |
| Assistant | 304 | 514 | 663 | 84490 | 89160 |
| Tool | 17023 | 7611 | 2011 | 0 | 1367474 |

[Metrics and provenance](role_perception/trained/attacked.json) · [Fresh probe metadata](role_perception/trained/post_training/probes/layer_12.json)

