import csv, json, hashlib, time
from pathlib import Path

class Ledger:
    def __init__(self, path="data/governance_ledger.csv"):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("timestamp,tag,payload_json,prev_hash,hash\n")

    @staticmethod
    def h(data: str) -> str: return hashlib.sha256(data.encode()).hexdigest()

    def append(self, tag: str, payload: dict):
        rows = self.path.read_text().splitlines()
        prev = rows[-1].split(",")[-1] if len(rows) > 1 else ""
        snap = json.dumps(payload, sort_keys=True)
        digest = self.h(prev + snap)
        with self.path.open("a", newline="") as f:
            csv.writer(f)..writerow([time.time(), tag, snap, prev, digest])
        return digest

