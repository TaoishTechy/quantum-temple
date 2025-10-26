from prometheus_client import Gauge, CollectorRegistry, generate_latest
registry = CollectorRegistry()

g_alignment = Gauge("qt_alignment_score","Ontology alignment score (0..1)", registry=registry)
g_norm_viol = Gauge("qt_norm_violations","Norm violations per step", registry=registry)
g_role_coh = Gauge("qt_role_coherence","Role coherence (0..1)", registry=registry)

def export_bytes():
    return generate_latest(registry)

def set_alignment(x): g_alignment.set(float(x))
def set_violations(n): g_norm_viol.set(float(n))
def set_role_coherence(x): g_role_coh.set(float(x))
