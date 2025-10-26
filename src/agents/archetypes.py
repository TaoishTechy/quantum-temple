from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re, yaml, pathlib

@dataclass
class ArchetypeSpec:
    name: str
    role: str
    traits: List[str]
    control: Dict[str, float]
    operators: List[str]

@dataclass
class NodeBinding:
    node_indices: List[int]
    archetype: ArchetypeSpec

class ArchetypeRegistry:
    def __init__(self):
        self.specs: Dict[str, ArchetypeSpec] = {}
        self.bindings: List[NodeBinding] = []

    @staticmethod
    def _parse_range(s: str) -> List[int]:
        if "-" in s:
            a, b = s.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(s)]

    def load_yaml(self, path: str | pathlib.Path):
        data = yaml.safe_load(pathlib.Path(path).read_text())
        self.specs.clear(); self.bindings.clear()
        for name, spec in data.get("archetypes", {}).items():
            self.specs[name] = ArchetypeSpec(
                name=name,
                role=spec.get("role",""),
                traits=list(spec.get("traits", [])),
                control=dict(spec.get("control", {})),
                operators=list(spec.get("operators", [])),
            )
        for key, at_name in data.get("binding", {}).get("nodes", {}).items():
            idxs = []
            for part in re.split(r"\s*,\s*", key):
                idxs.extend(self._parse_range(part))
            if at_name not in self.specs:
                raise KeyError(f"Unknown archetype: {at_name}")
            self.bindings.append(NodeBinding(node_indices=idxs, archetype=self.specs[at_name]))

    def get_archetype_for_node(self, idx: int) -> Optional[ArchetypeSpec]:
        for b in self.bindings:
            if idx in b.node_indices:
                return b.archetype
        return None
