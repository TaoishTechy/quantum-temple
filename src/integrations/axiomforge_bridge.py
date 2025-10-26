# Bridge AxiomForge → Quantum Temple
# Register symbolic operators (zeta/oracle/collatz) when AxiomForge is present
from typing import Protocol, Dict, Any
# Example imports once AxiomForge is installed:
# from axiomforge.operators import zeta_phase, oracle_sat, collatz_map

class SymbolicOp(Protocol):
    def __call__(self, state: "ResonanceState", **kw) -> "ResonanceState": ...

REGISTRY: Dict[str, SymbolicOp] = {
    # "zeta": zeta_phase,
    # "oracle": oracle_sat,
    # "collatz": collatz_map,
}

def get_op(name: str) -> SymbolicOp:
    if name not in REGISTRY:
        raise KeyError(f"Operator {name} not registered")
    return REGISTRY[name]
