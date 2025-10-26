from src.agents.archetypes import ArchetypeRegistry

def test_load_and_bind_archetypes(tmp_path):
    yml = tmp_path / "archetypes.yaml"
    yml.write_text("""
archetypes:
  A: { role: x, traits: [a], control: { sigma_q_target_variance: 0.05, parity_flip_budget: 1 }, operators: [zeta] }
binding:
  nodes:
    "0-2": A
""")
    reg = ArchetypeRegistry()
    reg.load_yaml(yml)
    assert "A" in reg.specs
    assert reg.get_archetype_for_node(0).name == "A"
    assert reg.get_archetype_for_node(2).name == "A"
    assert reg.get_archetype_for_node(3) is None
