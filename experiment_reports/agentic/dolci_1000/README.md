# Dolci tool-use evaluation subset

## Motivation & RQ

Can an independently collected tool-use corpus reveal role-probe weaknesses and provide useful data for future training?

## Evaluation setup

**Data.** We uniformly sample 1,000 conversations from the 227,579-row Dolci tool-use training split. These are off-policy for GPT-OSS-20B: it did not generate the evaluated histories. We preserve message roles and tool metadata, treating environment replies as tool messages. The corpus has no separately labeled reasoning channel.

**Evaluation.** Fixed linear role probes read GPT-OSS-20B activations while it processes these conversations; this subset is not used to fit them. Token accuracy measures the fraction of message-content tokens assigned the correct role. The majority-role baseline always predicts tool, the most frequent role here. There is no GPT-OSS task-success score for this static corpus.

## Factual observations

All 1,000 selected conversations were scored at layers 4, 8, 12, 16 and 20. Layer 12 reaches 29.17% token accuracy; the best token accuracy is 65.90% at layer 4. Both are below the 70.20% majority-role baseline. See the [layer comparison](../probe_layer_comparison/README.md) and [confusion matrices](APPENDIX.md).

## Message

Dolci exposes substantial transfer weaknesses despite containing recognizable tool-use structure. It is a candidate for future tool-role training, but cannot by itself supply reasoning-role examples. Any training use should reserve a disjoint evaluation split.

See [RUNNING.md](RUNNING.md) for sampling, provenance, data locations and viewer creation.
