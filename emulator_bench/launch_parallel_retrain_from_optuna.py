import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_LEVELS


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "emulator_bench" / "run_split_benchmarks.py"
DEFAULTS = REPO_ROOT / "emulator_bench" / "default_hparams_theia_original.json"


def hparams_for_level(base_dir: Path, level: str, template: str = "") -> Path:
    if template:
        candidate = Path(template.format(level=level))
    else:
        candidate = Path(base_dir) / "optuna_studies" / level / f"theia_{level}_optuna_best_hparams.json"
    if not candidate.exists():
        return DEFAULTS
    payload = json.loads(candidate.read_text())
    hparams = payload.get("best_hparams", payload)
    merged = json.loads(DEFAULTS.read_text())
    merged.update(hparams)
    output = candidate.parent / f"theia_{level}_merged_hparams.json"
    output.write_text(json.dumps(merged, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain Theia split jobs from Optuna best hparams.")
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--gpus", nargs="*", default=None)
    parser.add_argument("--hparams_json_template", default="")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for level in args.levels:
        hparams = hparams_for_level(Path(args.base_dir), level, args.hparams_json_template)
        cmd = [
            sys.executable,
            str(RUNNER),
            "--base_dir",
            args.base_dir,
            "--embeddings_dir",
            args.embeddings_dir,
            "--levels",
            level,
            "--hparams_json",
            str(hparams),
            *args.extra,
        ]
        if args.gpus:
            cmd.extend(["--gpus", *args.gpus])
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
