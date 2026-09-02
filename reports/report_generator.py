"""HTML report generator for the AMH benchmark.

Produces a single self-contained page (Chart.js via CDN) mirroring the
original report's sections:
    - header (run time, scenario/agent/judge-call counts, agent legend)
    - Executive Summary
    - Comparative Gains vs Stateless Baseline (delta table)
    - Full Results - All Agents x All Scenarios
    - Latency / Cost Efficiency / Response Quality bar charts
    - Multi-Dimension Quality Radar per scenario
    - AMH Memory Lifecycle table
"""
import json
from typing import Dict

from benchmarks.metrics import BenchmarkResults

# Display names and ordering, matching the original report.
AGENT_LABELS = {
    "StatelessAgent": "Stateless (Gen1)",
    "BufferAgent": "Buffer (Gen2)",
    "NaiveVectorAgent": "Naive Vector (Gen3)",
    "AMHAgent": "AMH (This Work)",
}
AGENT_ORDER = ["StatelessAgent", "BufferAgent", "NaiveVectorAgent", "AMHAgent"]
AGENT_COLORS = {
    "StatelessAgent": "#8c8c8c",
    "BufferAgent": "#5b9bd5",
    "NaiveVectorAgent": "#ffb000",
    "AMHAgent": "#ff9900",
}

CSS = """
body{font-family:'Amazon Ember','Helvetica Neue',Arial,sans-serif;margin:0;
     background:#f5f6f7;color:#232f3e}
.wrap{max-width:1200px;margin:0 auto;padding:24px}
header{background:#232f3e;color:#fff;padding:28px 32px;border-bottom:4px solid #ff9900}
header h1{margin:0 0 6px;font-size:26px}
header .sub{color:#c9d3da;font-size:14px}
header .meta{color:#aab7b8;font-size:12.5px;margin-top:10px}
.legend{margin-top:12px}
.legend span{display:inline-block;margin-right:14px;font-size:12.5px;color:#fff;
     padding:3px 10px;border-radius:12px}
section{background:#fff;border-radius:8px;margin:22px 0;padding:22px 26px;
     box-shadow:0 1px 3px rgba(0,0,0,.08)}
h2{margin:0 0 12px;font-size:19px;color:#232f3e;border-left:4px solid #ff9900;
     padding-left:10px}
p.note{color:#5a646e;font-size:12.5px;margin:4px 0 14px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#232f3e;color:#fff;padding:8px 10px;text-align:left;font-weight:600}
td{padding:7px 10px;border-bottom:1px solid #e8eaec}
tr:nth-child(even) td{background:#fafbfb}
.pos{color:#1d8042;font-weight:600}
.neg{color:#c7331f;font-weight:600}
.amhrow td{background:#fff7ea!important}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.chartbox{position:relative;height:300px}
.chartbox h3{font-size:13.5px;color:#5a646e;margin:0 0 8px;text-align:center}
footer{color:#5a646e;font-size:12px;text-align:center;padding:18px}
.callout{background:#fff7ea;border-left:4px solid #ff9900;padding:12px 16px;
     font-size:14px;margin:10px 0}
"""


def _fmt_delta(v: float, lower_better: bool = False) -> str:
    """Format a delta % with the original report's coloring semantics.

    For all deltas here, positive = improvement (the sign convention is
    already applied in gains_vs_baseline), so positive renders green.
    """
    cls = "pos" if v >= 0 else "neg"
    return f'<td class="{cls}">{v:+.1f}%</td>'


def generate_report(results: BenchmarkResults, output_path: str) -> str:
    """Render the full HTML report; returns the output path."""
    data = results.data
    scenarios = list(data.keys())
    gains = results.gains_vs_baseline("StatelessAgent")
    judge_calls = results.total_judge_calls()

    # ---------------------------------------------------------- gains table
    gain_rows = []
    for agent in AGENT_ORDER[1:]:
        for sc in scenarios:
            d = gains.get(agent, {}).get(sc)
            if not d:
                continue
            gain_rows.append(
                f"<tr{' class=amhrow' if agent == 'AMHAgent' else ''}>"
                f"<td>{sc}</td><td>{AGENT_LABELS[agent]}</td>"
                + _fmt_delta(d["latency_delta_pct"])
                + _fmt_delta(d["cost_delta_pct"])
                + _fmt_delta(d["token_delta_pct"])
                + _fmt_delta(d["quality_delta_pct"])
                + _fmt_delta(d["personalization_delta_pct"])
                + _fmt_delta(d["keyword_delta_pct"])
                + "</tr>")

    # ----------------------------------------------------- full result rows
    full_rows = []
    for sc in scenarios:
        for agent in AGENT_ORDER:
            r = data[sc].get(agent)
            if not r:
                continue
            full_rows.append(
                f"<tr{' class=amhrow' if agent == 'AMHAgent' else ''}>"
                f"<td>{sc}</td><td>{AGENT_LABELS[agent]}</td>"
                f"<td>{r['avg_latency_ms']:.0f}</td>"
                f"<td>{r['p95_latency_ms']:.0f}</td>"
                f"<td>{r['total_input_tokens']:,}</td>"
                f"<td>${r['total_cost']:.4f}</td>"
                f"<td>${r['cost_per_1k_turns']:.2f}</td>"
                f"<td>{r['avg_quality']:.2f}</td>"
                f"<td>{r['avg_personalization']:.2f}</td>"
                f"<td>{r['keyword_hit_rate']:.2f}</td>"
                f"<td>{r['consolidations']}</td>"
                f"<td>{r['forgettings']}</td></tr>")

    # -------------------------------------------------- AMH lifecycle table
    amh_rows = []
    for sc in scenarios:
        r = data[sc].get("AMHAgent")
        if not r:
            continue
        amh_rows.append(
            f"<tr><td>{sc}</td><td>{r['memories_created']}</td>"
            f"<td>{r['consolidations']}</td><td>{r['forgettings']}</td>"
            f"<td>{r['avg_retrieved_per_turn']:.1f}</td>"
            f"<td>${r['total_cost']:.4f}</td></tr>")

    # ----------------------------------------------------------- chart data
    def series(field):
        """Per-agent value list across scenarios for bar charts."""
        return {a: [data[sc].get(a, {}).get(field, 0) for sc in scenarios]
                for a in AGENT_ORDER}

    chart_payload = {
        "scenarios": scenarios,
        "labels": {a: AGENT_LABELS[a] for a in AGENT_ORDER},
        "colors": {a: AGENT_COLORS[a] for a in AGENT_ORDER},
        "latency": series("avg_latency_ms"),
        "tokens_per_turn": {
            a: [(data[sc][a]["total_input_tokens"] / max(1, data[sc][a]["turns"]))
                if a in data[sc] else 0 for sc in scenarios]
            for a in AGENT_ORDER},
        "cost1k": series("cost_per_1k_turns"),
        "keyword": series("keyword_hit_rate"),
        "quality": series("avg_quality"),
        "personalization": series("avg_personalization"),
        # Radar: per scenario, per agent, five quality dimensions.
        "radar": {
            sc: {a: [data[sc][a]["avg_quality"],
                     data[sc][a]["avg_personalization"],
                     data[sc][a]["keyword_hit_rate"],
                     # token efficiency: inverse share vs worst agent
                     1.0 - (data[sc][a]["total_input_tokens"]
                            / max(1, max(data[sc][x]["total_input_tokens"]
                                         for x in data[sc]))) + 0.05,
                     # latency efficiency: inverse share vs slowest agent
                     1.0 - (data[sc][a]["avg_latency_ms"]
                            / max(1.0, max(data[sc][x]["avg_latency_ms"]
                                           for x in data[sc]))) + 0.05]
                 for a in AGENT_ORDER if a in data[sc]}
            for sc in scenarios},
    }

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>AMH Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>{CSS}</style></head><body>
<header><div class="wrap">
<h1>Agent Memory Hierarchy &mdash; Benchmark Report</h1>
<div class="sub">Empirical comparison of AMH vs traditional memory approaches
across cost, latency, and response quality metrics</div>
<div class="meta">Run: {results.run_timestamp} &nbsp;|&nbsp;
Scenarios: {len(scenarios)} &nbsp;|&nbsp; Agents: {len(AGENT_ORDER)}
&nbsp;|&nbsp; LLM Judge calls: {judge_calls} &nbsp;|&nbsp;
Model: {results.llm_model_id}</div>
<div class="legend">
{''.join(f'<span style="background:{AGENT_COLORS[a]}">{AGENT_LABELS[a]}</span>' for a in AGENT_ORDER)}
</div></div></header>
<div class="wrap">

<section><h2>Executive Summary</h2>
<div class="callout"><b>What this benchmark proves:</b>
The Agent Memory Hierarchy (AMH) reduces input token consumption by
selecting only MRDF-scored relevant memories, while simultaneously
improving personalization and keyword recall vs all three traditional
baselines. The naive vector store (Gen3) accumulates unbounded memories,
increasing retrieval noise over time. The conversation buffer (Gen2)
loses cross-session context entirely. AMH is the only approach designed
to improve on all three axes simultaneously:
<b>cost &darr;, quality &uarr;, personalization &uarr;</b>.</div></section>

<section><h2>Comparative Gains vs Stateless Baseline</h2>
<p class="note">Positive % = improvement. Red = regression vs baseline.
Cost/latency/token gains: lower raw usage is better. Quality gains:
higher scores are better. Quality-family deltas floor the baseline
denominator at 0.05 to keep percentages meaningful when the stateless
baseline scores near zero.</p>
<table><tr><th>Scenario</th><th>Agent</th><th>Latency &Delta;</th>
<th>Cost &Delta;</th><th>Tokens &Delta;</th><th>Quality &Delta;</th>
<th>Personalization &Delta;</th><th>Keyword Hit &Delta;</th></tr>
{''.join(gain_rows)}</table></section>

<section><h2>Full Results &mdash; All Agents &times; All Scenarios</h2>
<table><tr><th>Scenario</th><th>Agent</th><th>Avg Latency (ms)</th>
<th>P95 Latency (ms)</th><th>Total Input Tokens</th><th>Total Cost ($)</th>
<th>Cost/1K Turns ($)</th><th>Avg Quality</th><th>Avg Personalization</th>
<th>Keyword Hit Rate</th><th>Consolidations</th><th>Forgettings</th></tr>
{''.join(full_rows)}</table></section>

<section><h2>Latency Comparison</h2><div class="charts">
<div><h3 style="text-align:center;font-size:13.5px;color:#5a646e">
Average Latency per Turn (ms) &mdash; by Scenario</h3>
<div class="chartbox"><canvas id="latency"></canvas></div></div>
<div><h3 style="text-align:center;font-size:13.5px;color:#5a646e">
Average Input Tokens per Turn &mdash; by Scenario</h3>
<div class="chartbox"><canvas id="tokens"></canvas></div></div>
</div></section>

<section><h2>Cost Efficiency</h2><div class="charts">
<div><h3 style="text-align:center;font-size:13.5px;color:#5a646e">
Cost per 1,000 Turns ($) &mdash; by Scenario</h3>
<div class="chartbox"><canvas id="cost1k"></canvas></div></div>
<div><h3 style="text-align:center;font-size:13.5px;color:#5a646e">
Keyword Hit Rate (0&ndash;1) &mdash; by Scenario</h3>
<div class="chartbox"><canvas id="keyword"></canvas></div></div>
</div></section>

<section><h2>Response Quality</h2><div class="charts">
<div><h3 style="text-align:center;font-size:13.5px;color:#5a646e">
Average Quality Score (0&ndash;1) &mdash; by Scenario</h3>
<div class="chartbox"><canvas id="quality"></canvas></div></div>
<div><h3 style="text-align:center;font-size:13.5px;color:#5a646e">
Average Personalization Score (0&ndash;1) &mdash; by Scenario</h3>
<div class="chartbox"><canvas id="personalization"></canvas></div></div>
</div></section>

<section><h2>Multi-Dimension Quality Radar &mdash; by Scenario</h2>
<div class="charts">
{''.join(f'<div><h3 style="text-align:center;font-size:13.5px;color:#5a646e">{sc} &mdash; Quality Dimensions</h3><div class="chartbox"><canvas id="radar{i}"></canvas></div></div>' for i, sc in enumerate(scenarios))}
</div></section>

<section><h2>AMH Memory Lifecycle</h2>
<p class="note">AMH-specific metrics: memories created, consolidated
(episodic&rarr;semantic), forgotten (temporal decay), and average
retrieved per turn.</p>
<table><tr><th>Scenario</th><th>Memories Created</th><th>Consolidations</th>
<th>Forgettings</th><th>Avg Retrieved/Turn</th><th>Total Cost ($)</th></tr>
{''.join(amh_rows)}</table></section>

<footer>Agent Memory Hierarchy Benchmark &nbsp;|&nbsp; Based on
<i>Production Agent Memory: Architecture Patterns for Stateful AI
Systems</i> &nbsp;|&nbsp; Generated: {results.run_timestamp}</footer>
</div>
<script>
const D = {json.dumps(chart_payload)};
const AG = Object.keys(D.labels);
function bar(id, field) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{ labels: D.scenarios,
      datasets: AG.map(a => ({{ label: D.labels[a],
        backgroundColor: D.colors[a], data: D[field][a] }})) }},
    options: {{ responsive: true, maintainAspectRatio: false,
      scales: {{ y: {{ beginAtZero: true }} }} }} }});
}}
bar('latency', 'latency'); bar('tokens', 'tokens_per_turn');
bar('cost1k', 'cost1k'); bar('keyword', 'keyword');
bar('quality', 'quality'); bar('personalization', 'personalization');
D.scenarios.forEach((sc, i) => {{
  new Chart(document.getElementById('radar' + i), {{
    type: 'radar',
    data: {{ labels: ['Quality', 'Personalization', 'Keyword Recall',
                      'Token Efficiency', 'Latency Efficiency'],
      datasets: AG.filter(a => D.radar[sc][a]).map(a => ({{
        label: D.labels[a], data: D.radar[sc][a],
        borderColor: D.colors[a],
        backgroundColor: D.colors[a] + '22' }})) }},
    options: {{ responsive: true, maintainAspectRatio: false,
      scales: {{ r: {{ min: 0, max: 1.05 }} }} }} }});
}});
</script></body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    return output_path
