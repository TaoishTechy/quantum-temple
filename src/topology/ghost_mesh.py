# src/topology/ghost_mesh.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class GhostMesh:
    n: int
    cross_links: Optional[List[Tuple[int,int]]] = None

    def neighbors(self) -> List[List[int]]:
        """Ring + mesh neighbors (undirected)."""
        nb = [[(i-1) % self.n, (i+1) % self.n] for i in range(self.n)]
        if self.cross_links:
            for a,b in self.cross_links:
                if b not in nb[a]: nb[a].append(b)
                if a not in nb[b]: nb[b].append(a)
        return nb
