"""
Draws the README's charts as SVG files from measured results, so that a figure can't drift from what was measured.
Only needs the standard library. They follow the viewer's light or dark setting.

    python training/make_charts.py
"""
import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "images"
RESULTS_DIR = ROOT / "docs" / "results"
STYLE = """<style>
.surface{fill:#fcfcfb}.title{fill:#0b0b0b}.ink{fill:#52514e}.grid{stroke:#e4e3df}.ring{stroke:#fcfcfb}
.s1{fill:#2a78d6;stroke:#2a78d6}.s2{fill:#eb6834;stroke:#eb6834}.s3{fill:#1baf7a;stroke:#1baf7a}
@media (prefers-color-scheme:dark){.surface{fill:#1a1a19}.title{fill:#f0efec}.ink{fill:#c3c2b7}.grid{stroke:#3a3936}.ring{stroke:#1a1a19}
.s1{fill:#3987e5;stroke:#3987e5}.s2{fill:#d95926;stroke:#d95926}.s3{fill:#199e70;stroke:#199e70}}
text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif}
</style>"""


def svg(width, height, body, description):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
            f'aria-label="{escape(description, quote=True)}">{STYLE}<rect class="surface" width="{width}" height="{height}" rx="8"/>{body}</svg>')


def text(x, y, content, kind="ink", size=13, anchor="start", weight=400):
    return f'<text class="{kind}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" stroke="none">{escape(str(content))}</text>'


def nice_top(value):
    """A round number just above value, for the top of an axis."""
    for step in (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        if value <= step * 5:
            return step * 5, step
    return value, value / 5


def legend(x, y, names):
    parts = []
    for index, name in enumerate(names, 1):
        parts.append(f'<rect class="s{index}" x="{x}" y="{y - 10}" width="12" height="12" rx="3" stroke="none"/>')
        parts.append(text(x + 18, y, name))
        x += 30 + 8 * len(name)
    return "".join(parts)


def training_chart(history):
    """Training and validation loss by epoch, with the epoch that was kept marked."""
    width, height, left, right, top, bottom = 720, 340, 56, 96, 74, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    epochs = [h["epoch"] for h in history]
    ceiling, step = nice_top(min(max(max(h["train"], h["validation"]) for h in history[2:]), 10.0))

    def place(epoch, value):
        return left + plot_w * (epoch - epochs[0]) / max(epochs[-1] - epochs[0], 1), top + plot_h * (1 - min(value, ceiling) / ceiling)

    body = [text(left, 30, "Validation stops improving well before training does", "title", 16, weight=600),
            text(left, 50, "Detector loss by epoch. The epoch with the lowest validation loss is the one kept.")]
    for tick in range(int(ceiling / step) + 1):
        y = top + plot_h * (1 - tick * step / ceiling)
        body.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke-width="1"/>')
        body.append(text(left - 8, y + 4, f"{tick * step:g}", anchor="end", size=12))
    for epoch in [e for e in epochs if e == epochs[0] or e % 10 == 0]:
        body.append(text(place(epoch, 0)[0], top + plot_h + 20, epoch, anchor="middle", size=12))
    body.append(text(left + plot_w / 2, height - 6, "Epoch", anchor="middle", size=12))
    for key, kind, label in (("train", "s1", "Training"), ("validation", "s2", "Validation")):
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (place(h["epoch"], h[key]) for h in history))
        body.append(f'<polyline class="{kind}" points="{points}" style="fill:none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        end = place(history[-1]["epoch"], history[-1][key])
        body.append(text(end[0] + 8, end[1] + 4, label))
    best = min(history, key=lambda h: h["validation"])
    x, y = place(best["epoch"], best["validation"])
    body.append(f'<circle class="s2 ring" cx="{x:.1f}" cy="{y:.1f}" r="5" stroke-width="2" style="stroke:var(--none)"/>')
    body.append(text(x, y - 12, f"Kept: epoch {best['epoch']}, {best['validation']:.2f}", anchor="middle", size=12))
    body.append(legend(left, height - 6, ("Training", "Validation")))
    return svg(width, height, "".join(body), f"Line chart of detector training and validation loss over {len(history)} epochs. The lowest validation loss, {best['validation']:.2f}, was at epoch {best['epoch']}.")


def comparison_chart(summary, modes, titles):
    """How close each hidden hazard came, by version of the car, as bars with collisions written beside them."""
    scenarios = [s for s in summary if any(summary[s].get(m, {}).get("closest_mean_m") is not None for m in modes)]
    band, gap, bar, left, right, top = 58, 22, 16, 150, 170, 84
    height = top + (band + gap) * len(scenarios) + 34
    width = 760
    plot_w = width - left - right
    largest = max(summary[s][m]["closest_mean_m"] or 0 for s in scenarios for m in modes if m in summary[s])
    ceiling, step = nice_top(max(largest, 0.5))
    body = [text(24, 30, "How close each hidden hazard came", "title", 16, weight=600),
            text(24, 50, "Mean closest distance to the hazard, in metres. Further is safer. 0 m is contact."),
            legend(left, 72, [titles[m] for m in modes])]
    for tick in range(int(ceiling / step) + 1):
        x = left + plot_w * tick * step / ceiling
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - 28}" stroke-width="1"/>')
        body.append(text(x, height - 10, f"{tick * step:g} m", anchor="middle", size=12))
    for row, scenario in enumerate(scenarios):
        y = top + 6 + row * (band + gap)
        body.append(text(left - 12, y + band / 2, titles.get(scenario, scenario), "title", 13, anchor="end", weight=500))
        for index, mode in enumerate(modes):
            result = summary[scenario].get(mode)
            if result is None:
                continue
            value = result["closest_mean_m"] or 0.0
            bar_y = y + index * (bar + 2)
            w = plot_w * value / ceiling
            if w >= 4:
                body.append(f'<path class="s{index + 1}" stroke="none" d="M{left},{bar_y} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bar - 8} a4,4 0 0 1 -4,4 h-{w - 4:.1f} z"/>')
            else:
                body.append(f'<rect class="s{index + 1}" stroke="none" x="{left}" y="{bar_y}" width="{max(w, 2):.1f}" height="{bar}"/>')
            hits = result["collisions"]
            label = f"{value:.2f} m" + (f", {hits} of {result['runs']} hit" if hits else "")
            body.append(text(left + w + 6, bar_y + 12, label, size=12))
    return svg(width, height, "".join(body), "Bar chart of the mean closest distance to each hidden hazard, for each version of the car. " +
               " ".join(f"{titles.get(s, s)}: " + ", ".join(f"{titles[m]} {summary[s][m]['closest_mean_m']} metres" for m in modes if m in summary[s]) + "." for s in scenarios))


def main():
    wrote = []
    history_path = ROOT / "models" / "rsu_detector.candidate.history.json"
    if not history_path.exists():
        history_path = RESULTS_DIR / "rsu_training_history.json"
    if history_path.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "rsu_training.svg").write_text(training_chart(json.loads(history_path.read_text())), encoding="utf-8")
        wrote.append("rsu_training.svg")
    comparison_path = RESULTS_DIR / "rsu_comparison.json"
    if comparison_path.exists():
        data = json.loads(comparison_path.read_text())
        titles = {"truth": "True positions", "rsu": "Roadside unit", "sight": "Own sight only", "runner": "Red-light runner", "pedestrian": "Hidden pedestrian", "turner": "Oncoming turner", "clear": "Clear road"}
        modes = [m for m in ("sight", "rsu", "truth") if any(m in v for v in data["summary"].values())]
        (OUT_DIR / "rsu_comparison.svg").write_text(comparison_chart(data["summary"], modes, titles), encoding="utf-8")
        wrote.append("rsu_comparison.svg")
    print("Wrote", ", ".join(wrote) if wrote else "nothing: no results yet")


if __name__ == "__main__":
    sys.exit(main())
