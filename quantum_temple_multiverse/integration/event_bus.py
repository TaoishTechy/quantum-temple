"""
FILE: quantum_temple_multiverse/integration/event_bus.py
PURPOSE: Typed pub/sub for cross-module communication within/among realities.
MATHEMATICAL CORE: N/A (coordination primitive), includes backpressure cap.
INTEGRATION POINTS: multiverse.core, entities.*, cognition.*, civilizations.*, expansion.*
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, Deque, Any, List

Handler = Callable[[Dict[str, Any]], None]

@dataclass
class EventBus:
    max_queue: int = 2048

    def __post_init__(self):
        self.subs: Dict[str, List[Handler]] = defaultdict(list)
        self.q: Deque[Dict[str, Any]] = deque()

    def subscribe(self, topic: str, fn: Handler) -> None:
        self.subs[topic].append(fn)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if len(self.q) >= self.max_queue:
            # backpressure: drop oldest to keep system responsive
            self.q.popleft()
        self.q.append({"topic": topic, "payload": payload})

    def drain(self, max_events: int = 128) -> int:
        n = 0
        while self.q and n < max_events:
            ev = self.q.popleft()
            for fn in self.subs.get(ev["topic"], []):
                try:
                    fn(ev["payload"])
                except Exception:
                    # never crash the bus
                    pass
            n += 1
        return n

if __name__ == "__main__":
    bus = EventBus(max_queue=8)
    hits = []
    bus.subscribe("reality.tick", lambda e: hits.append(e["rid"]))
    for i in range(5):
        bus.publish("reality.tick", {"rid": f"R{i+1}"})
    print("drained:", bus.drain(), "hits:", hits)
