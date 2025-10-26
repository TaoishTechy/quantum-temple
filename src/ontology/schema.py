from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

Role = Literal["Agent","Intention","Constraint","Value","Evidence","Context","Operator","Signal","Process"]

class Concept(BaseModel):
    id: str
    label: str
    role: Role
    props: Dict[str, float] = Field(default_factory=dict)

class Relation(BaseModel):
    s: str      # subject id
    p: str      # predicate (e.g., "supports", "violates", "targets")
    o: str      # object id
    weight: float = 1.0

class Ontology(BaseModel):
    concepts: Dict[str, Concept] = Field(default_factory=dict)
    relations: List[Relation]     = Field(default_factory=list)

    def add(self, c: Concept): self.concepts[c.id] = c
    def link(self, s: str, p: str, o: str, w: float=1.0):
        self.relations.append(Relation(s=s, p=p, o=o, weight=w))
