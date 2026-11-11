# Role probes

The training design and GPT-OSS headers follow
[Prompt Injection as Role Confusion](https://github.com/role-confusion/prompt-injection-as-role-confusion)
(commit `ec333c40fd43fe991e1ebf66765051b6d7e35784`). Its license is included here.

- `model.py`: frozen GPT-OSS forward passes and pre-MLP activation capture.
- `linear.py`: the probe fitters (cuML logistic regression, the training_against_probes
  AdamW fit) and portable NPZ/JSON artifacts. Prediction needs only NumPy/SciPy; cuML is
  imported only during GPU fitting.
- `evaluation.py`, `audit.py`: token metrics and the cuML objective check.

Rendering and content-token labels come from `role_probe_training.harmony`. A probe has
one class per role of its role set, listed under `roles` in its metadata:

- the five roles (`ROLES`): `cot` means an assistant message with channel `analysis`;
  `assistant` means final or commentary, including tool calls;
- the header roles (`HEADER_ROLES`): one role per Harmony header form, which separate the
  system from the developer message, commentary text, final answers and tool calls from
  each other, and tool calls and tool outputs on the analysis channel from those on the
  commentary channel.

[HARMONY_RENDERING.md](../../../HARMONY_RENDERING.md) section 6 defines both. No reasoning is
inferred from the prose. Arbitrary trajectory formats and other models are not supported yet.
