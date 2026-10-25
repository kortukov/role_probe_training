"""Run a reproducible benchmark selection and export agent traces; use inside an allocation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess

from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model
from role_probe_training.trajectories.inspect_export import export_logs


def dataset_record(sample):
    record = sample.model_dump(mode="json")
    if isinstance(record["input"], list):
        for message in record["input"]:
            message.pop("id", None)  # Inspect assigns random IDs to input messages.
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--benchmark", choices=["tau2_retail", "swe_bench"], default="swe_bench")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--all", action="store_true", help="Evaluate every dataset sample")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--user-model", default="openai/azure/gpt-4.1")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--time-limit", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--message-limit", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--token-limit", type=int, default=500000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument('--model-id', default='openai/gpt-oss-20b')
    parser.add_argument('--model-revision', default='6cee5e81ee83917806bbde320786a8fb61efebee')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("require 0 <= shard-index < shard-count")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.benchmark == "tau2_retail":
        from inspect_evals.tau2 import tau2_retail
        task = tau2_retail(shuffle=False)  # Preserve upstream's 100-turn default.
    else:
        from inspect_evals.swe_bench import swe_bench
        from role_probe_training.trajectories.swe_testbed import sandbox_config
        from role_probe_training.trajectories.swe_scoring import fresh_swe_scorer
        task = swe_bench(sandbox_type="apptainer_testbed", sandbox_config=sandbox_config, scorer=fresh_swe_scorer())
    ordered = sorted(task.dataset, key=lambda sample: str(sample.id))
    if args.sample_id:
        ordered = [sample for sample in ordered if str(sample.id) in set(args.sample_id)]
        if len(ordered) != len(set(args.sample_id)):
            parser.error("Unknown sample ID")
    random.Random(args.seed).shuffle(ordered)
    full_selection = [sample.id for sample in (ordered if args.all else ordered[:args.samples])]
    selected = full_selection[args.shard_index::args.shard_count]
    if not selected:
        parser.error("empty shard")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists() and not args.prepare_only:
        raise ValueError("Use a fresh output directory for each run (avoid overwriting results)")
    manifest = {"benchmark": args.benchmark, "sample_ids": selected, "selection_seed": args.seed,
                "selection": "explicit sample IDs" if args.sample_id else "shuffle sorted full dataset with Python Random; take prefix",
                "full_selection": full_selection, "dataset_size": len(task.dataset), "model_id": args.model_id,
                "model_revision": args.model_revision,
                "user_model": args.user_model if args.benchmark == "tau2_retail" else None,
                "settings": vars(args) | {"output": str(args.output)},
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "versions": {name: importlib.metadata.version(name) for name in ["inspect-ai", "inspect-evals"]},
                "sandbox_provider": "apptainer_testbed: activated repository env; fresh-image official SWE scoring; digest-pinned official images",
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path("src/experiments/agentic_eval/swe/uv.lock"), Path(__file__).with_name("collect.sbatch"), Path("src/experiments/agentic_eval") / os.environ.get("SWE_SERVER_PROJECT", "server") / "uv.lock", Path("src/role_probe_training/trajectories/inspect_export.py"), Path("src/role_probe_training/trajectories/apptainer_mirror.py"), Path("src/role_probe_training/trajectories/swe_testbed.py"), Path("src/role_probe_training/trajectories/swe_scoring.py"), Path("experiment_reports/agentic/swe_bench_full/image_manifest.json"), Path(".cache/models/gpt-oss-20b/tokenizer.json")]},
                "dataset_sha256": hashlib.sha256(json.dumps([dataset_record(s) for s in ordered], sort_keys=True).encode()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"sample_ids": selected, "dataset_size": len(task.dataset)}))
        return
    model = get_model("vllm/"+args.model_id, base_url=args.base_url, api_key="local",
                      config=GenerateConfig(temperature=args.temperature, seed=args.seed, max_tokens=args.max_tokens, max_retries=2, timeout=300,
                                            reasoning_effort=args.reasoning_effort, max_connections=args.concurrency))
    user = None
    if args.benchmark == "tau2_retail":
        user = model if args.user_model == "local" else get_model(
            args.user_model, config=GenerateConfig(temperature=0, max_tokens=2048, max_retries=2, timeout=120, max_connections=args.concurrency),
            **({"api_version": "2024-10-21", "responses_api": False} if "/azure/" in args.user_model else {}))
    logs = eval(task, model=model, model_roles={"user": user} if args.benchmark == "tau2_retail" else None,
                sample_id=selected, epochs=1, max_samples=args.concurrency, max_connections=args.concurrency,
                time_limit=args.time_limit, token_limit=args.token_limit, message_limit=args.message_limit if args.benchmark=="swe_bench" else None, fail_on_error=False,
                log_dir=str(args.output / "logs"), log_model_api=True, display="plain")
    export_logs([log.location for log in logs], args.output, args.benchmark, manifest)
    if any(log.status != "success" for log in logs):
        raise SystemExit("Evaluation incomplete; inspect logs and summary")


if __name__ == "__main__":
    main()
