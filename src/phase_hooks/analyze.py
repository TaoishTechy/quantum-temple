import json, sys
out = "data/output.json"
if "--export" in sys.argv:
    out = sys.argv[sys.argv.index("--export")+1]
print("Analyze hook: exporting final report to", out)
with open(out, "w") as f:
    json.dump({"status": "ok", "message": "Quantum Temple analysis complete"}, f)
