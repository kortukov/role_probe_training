"""Publish complete fresh-rollout results, preserving the authored report text."""

import argparse
import hashlib
import json
from pathlib import Path


BENCHMARKS = {
    "appworld_test": ("AppWorld test_normal", 168),
    "tau2_retail": ("τ² retail", 114),
    "swe_bench": ("SWE-bench Verified", 500),
}
ROLE_NAMES = ["System", "User", "Reasoning", "Assistant", "Tool"]


def section(path, name, body, before=None):
    text = path.read_text()
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    block = f"{start}\n{body.strip()}\n{end}\n"
    if start in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        text = prefix + block.rstrip("\n") + suffix
    elif before:
        assert before in text, f"Missing insertion point in {path}"
        text = text.replace(before, block + "\n" + before, 1)
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text)


def publish(root, native_swe=False):
    tasks = json.loads((root / "post_training/task_results.json").read_text())
    export = json.loads((root / "data/post_training/model/export.json").read_text())
    probe = json.loads((root / "post_training/probes/layer_12.json").read_text())
    assert probe["fresh_fit"] and probe["layer"] == 12
    assert probe["model_revision"] == export["adapter_sha256"]
    results = {}
    table = [
        "### Fresh-trajectory evaluation\n",
        "| Benchmark | Task completion | Evaluation errors | Role accuracy | Macro recall |",
        "|---|---:|---:|---:|---:|",
    ]
    appendix = [
        "## Fresh-trajectory confusions\n",
        "Each row is a ground-truth content role and each column a predicted role. "
        "Values are percentages of tokens in that row; rounding may affect totals. "
        "A dash denotes a role absent from the cohort. Probe scores include only whole histories within native context; task scores include every attempt.\n",
    ]
    for key, (label, expected) in BENCHMARKS.items():
        task = tasks[key]
        prediction = root / f"predictions/post_training/{key}"
        if key == "tau2_retail":
            prediction = root / "predictions/post_training_whole/tau2_retail"
        if native_swe and key == "swe_bench":
            prediction = root / "predictions/swe_native"
            task = json.loads((root / "post_training/swe_native_task_results.json").read_text())
        score = json.loads((prediction / "summary.json").read_text())
        assert task["samples"] == score["trajectories"] == expected
        assert score["input_sha256"] == task["input_sha256"]
        assert score["model_revision"] == export["adapter_sha256"]
        assert score["probe_sha256"] == hashlib.sha256((root / "post_training/probes/layer_12.npz").read_bytes()).hexdigest()
        assert len(score["confusion_counts"]) == 5
        assert all(len(row) == 5 and all(n >= 0 for n in row) for row in score["confusion_counts"])
        total = sum(map(sum, score["confusion_counts"]))
        assert total > 0
        assert abs(sum(score["confusion_counts"][i][i] for i in range(5)) / total - score["accuracy"]) < 1e-10
        assert score["roles"] == ["system", "user", "cot", "assistant", "tool"]
        # Benchmarks may expose several diagnostics; use their task-success scorer.
        preferred = {"appworld_test": "appworld_success", "tau2_retail": "retail_scorer", "swe_bench": "swe_bench_scorer"}[key]
        available = task["metrics"]
        if preferred in available:
            metric = available[preferred]
        elif len(available) == 1:
            preferred, metric = next(iter(available.items()))
        else:
            raise ValueError(f"Select the task-success metric explicitly: {key}: {list(available)}")
        assert metric["total"] == expected
        table.append(
            f"| {label} | {metric['successes']}/{expected} ({metric['success_rate']:.1%}) "
            f"| {task['sample_errors']} | {score['accuracy']:.2%} | {score['macro_recall']:.2%} |"
        )
        appendix.extend([
            f"\n### {label}\n",
            "| True / predicted | " + " | ".join(ROLE_NAMES) + " |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, counts in zip(ROLE_NAMES, score["confusion_counts"]):
            total = sum(counts)
            values = [f"{100 * count / total:.2f}%" if total else "—" for count in counts]
            appendix.append("| " + name + " | " + " | ".join(values) + " |")
        excluded = score.get("excluded_trajectories", [])
        assert not score.get("segmented_trajectories"), "Chunked scores cannot enter this report"
        scored = score.get("scored_trajectories", expected)
        assert scored + len(excluded) == expected
        appendix.append(f"\nProbe coverage: {scored}/{expected} whole histories. "
                        f"{len(excluded)} exceed native context and have no probe score; "
                        "task completion still includes every attempt.")
        results[key] = {"task": task, "probe": score, "task_metric": preferred}
    table.extend([
        "\nThese are new trajectories generated by the final trained model, evaluated "
        "with an independently fitted layer-12 probe. They test role recovery on the "
        "model’s own behavior, separately from replaying the fixed development histories. "
        "[Role confusion matrices](APPENDIX.md#fresh-trajectory-confusions) are in the appendix. "
        "SWE generation used a 32K serving limit; context exhaustion constrains these task results "
        "([comparison and failure audit](" + ("APPENDIX.md#swe-bench-failure-audit" if root.name == "training_against_role_probes" else "../APPENDIX.md#swe-bench-failure-audit") + "))."
    ])
    if native_swe:
        table[-1] = table[-1].split("SWE generation used")[0] + "SWE uses the native 131K serving allowance with matched base/trained settings; see the [four-model comparison](../README.md)."
    section(root / "README.md", "fresh-rollout-results", "\n".join(table), before="## Message")
    section(root / "APPENDIX.md", "fresh-rollout-confusions", "\n".join(appendix))
    (root / "post_training/results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    publish(parser.parse_args().root)
