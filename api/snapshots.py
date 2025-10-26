from fastapi import APIRouter
import json, pathlib

router = APIRouter()

@router.get("/snapshot/maintenance")
def maintenance_snapshot():
    p = pathlib.Path("data/snapshots/maintenance_report.json")
    if not p.exists():
        return {"error": "snapshot not found"}
    return json.loads(p.read_text())
