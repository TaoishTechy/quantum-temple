from .schema import Ontology, Concept
from typing import Dict

class Binding:
    """
    Connects numerical runtime (nodes/operators) with semantic objects.
    Example: operator "H_obs" targets Intention "Stability", constrained by "Safety".
    """
    def __init__(self, onto: Ontology):
        self.onto = onto
        self.operator_map: Dict[str, str] = {}   # op_name -> concept_id
        self.node_roles: Dict[int, str]   = {}   # node_idx -> role (Agent/Signal/Process)

    def register_operator(self, op_name: str, concept_id: str):
        assert concept_id in self.onto.concepts
        self.operator_map[op_name] = concept_id

    def set_node_role(self, idx: int, role: str):
        self.node_roles[idx] = role

    def link_operator_to_intention(self, op_name: str, intention_id: str, weight=1.0):
        self.onto.link(self.operator_map[op_name], "targets", intention_id, w=weight)

    def link_operator_to_constraint(self, op_name: str, constraint_id: str, weight=1.0):
        self.onto.link(self.operator_map[op_name], "fulfills", constraint_id, w=weight)
