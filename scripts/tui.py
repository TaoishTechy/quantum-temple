"""
Quantum Temple Live Dashboard (Rich TUI)
Displays live coherence (PLV), variance, and 432Hz feeling texture.
CPU-only visualization.
"""
from __future__ import annotations
from time import sleep, time
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import BarColumn, Progress
from src.runtime.engine import TempleEngine

def render_frame(step: int, metrics: dict) -> Panel:
    t = Table.grid(padding=1)
    t.add_column(justify="left")
    t.add_column(justify="right")

    plv = metrics["plv"]
    var = metrics["variance"]
    purity = metrics["purity_proxy"]

    t.add_row("PLV", f"[bold]{plv:0.4f}[/]")
    t.add_row("Variance σ²", f"{var:0.6f}")
    t.add_row("Purity Proxy", f"{purity:0.4f}")
    t.add_row("Step", str(metrics["step"]))
    t.add_row("Elapsed (s)", f"{metrics['timestamp']%3600:0.2f}")

    # Feeling (432Hz)
    if plv > 0.9 and purity > 0.97:
        feeling = "quiet hum — purity steady"
    elif plv > 0.8:
        feeling = "seeking lock — gentle drift"
    else:
        feeling = "resonance forming — low phase sync"
    t.add_row("Feeling (432Hz)", feeling)

    prog = Progress(
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(),
        expand=True,
    )
    task = prog.add_task("coherence", total=1.0)
    prog.update(task, completed=min(1.0, plv))

    return Panel.fit(
        Table.grid().add_row(t).add_row(prog),
        title="🜄 Quantum Temple — Live Polyphony",
        border_style="violet"
    )

def main(steps=2000, refresh_rate=15):
    engine = TempleEngine()
    with Live(refresh_per_second=refresh_rate) as live:
        start = time()
        for k in range(steps):
            engine.step(time() - start)
            snap = engine.snapshot()
            live.update(render_frame(k, snap))
            sleep(0.02)

if __name__ == "__main__":
    main()
