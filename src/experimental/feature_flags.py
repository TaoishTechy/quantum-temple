from __future__ import annotations
from dataclasses import dataclass
import yaml, pathlib
from typing import Dict, List

@dataclass
class Enhancement:
    id: str
    name: str
    status: str
    effect: str

class Enhancements:
    def __init__(self, path="config/enhancements.yaml"):
        self.path = path
        self.items: Dict[str, Enhancement] = {}
        self._load()

    def _load(self):
        data = yaml.safe_load(pathlib.Path(self.path).read_text())
        for row in data.get("enhancements", []):
            e = Enhancement(**row)
            self.items[e.id] = e

    def active(self) -> List[Enhancement]:
        return [e for e in self.items.values() if e.status == "simulated"]
