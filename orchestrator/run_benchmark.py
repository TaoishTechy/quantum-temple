import json, os, subprocess, time, pathlib, hashlib

def commit_hash():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def run():
    run_id = time.strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path("data/benchmarks") / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "commit": commit_hash(),
        "ts": run_id,
        "env": {
            "NODES": os.getenv("NODES","8"),
            "INSTANCE_TYPE": os.getenv("INSTANCE_TYPE","dev-local"),
            "PYTHONHASHSEED": os.getenv("PYTHONHASHSEED","0")
        }
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    # run Blitz
    subprocess.check_call(["./orchestrator/run_blitz_test.sh"], env={**os.environ, "PYTHONHASHSEED":"0"})

    # Copy latest run logs (produced by blitz)
    logs_root = pathlib.Path("data/output_logs")
    latest = sorted(logs_root.glob("*"))[-1] if logs_root.exists() else None
    if latest:
        subprocess.check_call(["cp","-r", str(latest), str(outdir / "blitz_logs")])

    print(f"Benchmark captured: {outdir}")

if __name__ == "__main__":
    run()
