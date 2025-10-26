from .schema import Ontology

def to_jsonld(onto: Ontology) -> dict:
    ctx = {"@vocab": "https://ghost-mesh.io/ontology#"}
    graph = []
    for c in onto.concepts.values():
        graph.append({"@id": c.id, "label": c.label, "role": c.role, **c.props})
    for r in onto.relations:
        graph.append({"@id": f"{r.s}-{r.p}-{r.o}", "subject": r.s, "predicate": r.p, "object": r.o, "weight": r.weight})
    return {"@context": ctx, "@graph": graph}
