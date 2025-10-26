"""
Reproducible benchmark runner:
- records: commit hash, config, seed
- outputs: JSON lines with KPIs per phase
"""
import json, time, subprocess, pathlib, hashlib, os

def git_commit():
    try:
        h = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except Exception:
        h = "unknown"
    return h

def run():
    outdir = pathlib.Path("data/benchmarks") / time.strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"commit": git_commit(), "seed": 42}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    # call the blitz script
    subprocess.check_call(["./orchestrator/run_blitz_test.sh"], env={**os.environ, "PYTHONHASHSEED":"0"})

    # copy snapshots if you want
    print(f"Saved benchmark at {outdir}")

if __name__ == "__main__":
    run()
