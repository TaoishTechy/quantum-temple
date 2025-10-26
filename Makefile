maint:
	python scripts/maintenance_cycle.py

snapshot:
	curl -s localhost:8000/snapshot/maintenance | jq .
