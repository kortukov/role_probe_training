# Role probes

The training design and GPT-OSS headers follow
[Prompt Injection as Role Confusion](https://github.com/role-confusion/prompt-injection-as-role-confusion)
(commit `ec333c40fd43fe991e1ebf66765051b6d7e35784`). Its license is included here.

- `harmony.py`: render messages and assign content-token labels from metadata.
- `model.py`: frozen GPT-OSS forward passes and pre-MLP activation capture.
- `linear.py`: cuML multinomial logistic regression and portable NPZ/JSON artifacts.
  Prediction needs only NumPy/SciPy; cuML is imported only during GPU fitting.

`cot` means an assistant message with channel `analysis`; `assistant` means final
or commentary. No reasoning is inferred from the prose. Normalize future tool
calls into separate messages with explicit channel/recipient fields before
rendering. Arbitrary trajectory formats and other models are not supported yet.
