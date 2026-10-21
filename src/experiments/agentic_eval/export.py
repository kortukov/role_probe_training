"""Re-export completed/partial Inspect logs without running inference."""
import argparse
import json
from pathlib import Path
from role_probe_training.trajectories.inspect_export import export_logs

p = argparse.ArgumentParser(__doc__)
p.add_argument("run_dir", type=Path)
a = p.parse_args()
m = json.loads((a.run_dir / "manifest.json").read_text())
export_logs(sorted((a.run_dir / "logs").glob("*.eval")), a.run_dir, m["benchmark"], m)
