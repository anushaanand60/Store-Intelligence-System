import time

import requests
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table


def generate_dashboard_layout(metrics, funnel) -> Layout:
    layout = Layout()
    layout.split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1)
    )

    m_table = Table(title="🛒 Live Store Telemetry", expand=True)
    m_table.add_column("Metric Title", style="cyan", justify="left")
    m_table.add_column("Operational Value", style="magenta", justify="right")
    m_table.add_row("Unique Customer Count", str(metrics.get("unique_visitors", 0)))
    m_table.add_row("Live Store Occupancy", str(metrics.get("live_occupancy", 0)))
    m_table.add_row("Total Entry Detections", str(metrics.get("total_entries", 0)))
    m_table.add_row("Total Exit Detections", str(metrics.get("total_exits", 0)))
    m_table.add_row("Deduplicated Staff On-Floor", str(metrics.get("hourly_staff_counts", 0)))
    m_table.add_row("Offline Conversion Rate", f"{metrics.get('store_conversion_rate_percentage', 0)}%")
    m_table.add_row("Max Billing Queue Depth", str(metrics.get("queue_depth", 0)))
    m_table.add_row("Checkout Abandonment Rate", f"{metrics.get('abandonment_rate', 0)}%")

    f_table = Table(title="📊 Sequential Journey Funnel Drop-offs", expand=True)
    f_table.add_column("Conversion Stage", style="green", justify="left")
    f_table.add_column("Session Unit Count", style="yellow", justify="right")
    f_table.add_column("Step Drop-off %", style="red", justify="right")
    for stage in funnel:
        f_table.add_row(
            stage.get("stage", "Unknown"),
            str(stage.get("count", 0)),
            f"{stage.get('drop_off_percentage', 0.0)}%"
        )

    layout["left"].update(Panel(m_table, border_style="blue", title="[🔥 REAL-TIME STATE METRICS]"))
    layout["right"].update(Panel(f_table, border_style="green", title="[📉 CONVERSION DROPS]"))
    return layout


def run_dashboard():
    url = "http://localhost:8000/stores/STORE_BLR_002"
    with Live(refresh_per_second=1, screen=True) as live:
        while True:
            try:
                m = requests.get(f"{url}/metrics", timeout=2).json()
                f = requests.get(f"{url}/funnel", timeout=2).json()
                live.update(generate_dashboard_layout(m, f))
            except Exception:
                live.update(Panel(
                    "\n[bold red]CRITICAL BOUNDARY ERROR:[/bold red]\n"
                    "Awaiting stream packet ingestion or central api matrix runtime connection...",
                    title="[⚠️ SYSTEM STANDBY]",
                    border_style="yellow"
                ))
            time.sleep(1)

if __name__ == "__main__":
    run_dashboard()
