"""Pareto-frontier cost-vs-accuracy chart for Athanor on ARC-AGI-2 public eval.

Reads `hf_per_config.json` (produced by `aggregate_hf_corpus.py`) alongside this
script and writes the chart to `docs/images/cost_vs_accuracy.png` at the repo root.

Usage:
    python tools/plot_cost_vs_accuracy.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
JSON_PATH  = SCRIPT_DIR / "hf_per_config.json"
OUT_PATH   = REPO_ROOT / "docs" / "images" / "cost_vs_accuracy.png"

BG     = "#0a0a0a"
TEXT   = "#ffffff"
MUTED  = "#888888"
AGENT  = "#b57082"   # other agentic — muted dusty rose (desaturated so it doesn't compete with mint Athanor)
ACCENT = "#00e676"   # Athanor — electric mint (brighter than stock green)
FRONT  = "#ffffff"   # frontier line

# Vendor palette for HF CoT-only dots (distinct from Athanor mint and agentic coral)
V_ANTHROPIC = "#d97757"  # warm orange
V_OPENAI    = "#4a9eff"  # blue
V_GEMINI    = "#9ccc65"  # leaf green (distinct from mint)
V_GROK      = "#b188e5"  # violet
V_OTHER     = "#78909c"  # slate

def _vendor(tid):
    t = tid.lower()
    if any(k in t for k in ("claude","opus","sonnet","haiku")): return V_ANTHROPIC
    if any(k in t for k in ("gpt","o3")):                       return V_OPENAI
    if "gemini" in t:                                           return V_GEMINI
    if "grok" in t:                                             return V_GROK
    return V_OTHER

with open(JSON_PATH) as f:
    hf_configs = json.load(f)

agentic = [
    # (name, cost, score, is_athanor)
    ("Confluence",                    11.77, 97.9, False),
    ("Squeeze-Evolve",                 5.93, 97.5, False),
    ("Imbue DE",                       8.71, 95.1, False),
    ("Imbue DE (Flash)",               2.42, 61.4, False),
    ("Athanor (this work)",            3.12, 95.7, True),
]

# Build combined point set: (cost, score, label, kind)
points = []
for c in hf_configs:
    points.append((c["cost_per_task"], c["task_accuracy_pct"], c["test_id"], "hf"))
for name, cost, score, is_a in agentic:
    points.append((cost, score, name, "athanor" if is_a else "agentic"))

# Pareto frontier: point is on frontier if no other point has lower cost AND higher accuracy
def on_frontier(p, all_pts):
    c, s, _, _ = p
    for q in all_pts:
        qc, qs, _, _ = q
        if qc <= c and qs >= s and (qc < c or qs > s):
            return False
    return True

frontier_pts = [p for p in points if on_frontier(p, points)]
frontier_pts.sort(key=lambda x: x[0])

fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=BG)
ax.set_facecolor(BG)

# 95% threshold band
ax.axhspan(95, 100, facecolor="#ffffff", alpha=0.04, zorder=1)
ax.axhline(95, color="#555", linestyle=":", linewidth=1, alpha=0.5, zorder=2)

# Per-family sweep lines (arcprize-style: connect same-model variants to show cost/effort sweep)
def _family(tid):
    if "claude-opus-4-6-thinking-120K" in tid: return "opus-4-6"
    if "claude-opus-4-5" in tid and "thinking" in tid: return "opus-4-5"
    if "gpt-5-2" in tid and "thinking" in tid: return "gpt-5-2-thinking"
    if "gpt-5-2-pro" in tid: return "gpt-5-2-pro"
    if "gemini-3-flash-preview-thinking" in tid: return "gemini-3-flash"
    if "gpt-5-4-nano" in tid: return "gpt-5-4-nano"
    if "gpt-5-1" in tid and "thinking" in tid: return "gpt-5-1-thinking"
    if "claude-haiku-4-5-20251001" in tid: return "haiku-4-5"
    if "claude-sonnet-4-5-20250929" in tid: return "sonnet-4-5"
    return None

fam_color = {
    "opus-4-6":        "#d97757",  # Anthropic warm orange
    "opus-4-5":        "#d97757",
    "haiku-4-5":       "#d97757",
    "sonnet-4-5":      "#d97757",
    "gpt-5-2-thinking":"#4a9eff",  # OpenAI blue
    "gpt-5-2-pro":     "#4a9eff",
    "gpt-5-1-thinking":"#4a9eff",
    "gpt-5-4-nano":    "#4a9eff",
    "gemini-3-flash":  "#9ccc65",  # Gemini leaf green (distinct from Athanor mint)
}

fam_pts = defaultdict(list)
for c in hf_configs:
    f = _family(c["test_id"])
    if f is None: continue
    fam_pts[f].append((c["cost_per_task"], c["task_accuracy_pct"]))
for fname, pts in fam_pts.items():
    pts.sort()
    # Keep only points within the plot's x-range so lines don't trail off-canvas
    vis = [(x, y) for x, y in pts if 0 <= x <= 14]
    if len(vis) < 2: continue
    xs = [p[0] for p in vis]; ys = [p[1] for p in vis]
    ax.plot(xs, ys, color=fam_color.get(fname, "#888"),
            alpha=0.50, linewidth=1.2, linestyle="--", zorder=2.5)

# Pareto frontier — bold dashed line connecting frontier points
fc = [p[0] for p in frontier_pts]
fs = [p[1] for p in frontier_pts]
ax.plot(fc, fs, color=FRONT, alpha=0.55, linewidth=2.0,
        linestyle="--", zorder=3, label="Pareto frontier")

# All HF dots plotted individually, colored by vendor family
frontier_ids = {id(p) for p in frontier_pts}
hf_non_fr_x, hf_non_fr_y, hf_non_fr_c = [], [], []
hf_fr_x, hf_fr_y, hf_fr_c = [], [], []
for p in points:
    if p[3] != "hf":
        continue
    c, s, tid, _ = p
    vc = _vendor(tid)
    if id(p) in frontier_ids:
        hf_fr_x.append(c); hf_fr_y.append(s); hf_fr_c.append(vc)
    else:
        hf_non_fr_x.append(c); hf_non_fr_y.append(s); hf_non_fr_c.append(vc)

DOT_SIZE = 110       # agentic + HF frontier
DOM_SIZE = 50        # dominated HF — smaller to reduce overlap
ATHANOR_SIZE = 180   # smaller triangle — glow provides thumbnail visibility; small size makes 95%-line position unambiguous
# Dominated: vendor color, muted alpha
ax.scatter(hf_non_fr_x, hf_non_fr_y, s=DOM_SIZE, c=hf_non_fr_c, alpha=0.70,
           edgecolors="none", zorder=4)
# Frontier: vendor color, fully saturated, white halo to mark frontier
ax.scatter(hf_fr_x, hf_fr_y, s=DOT_SIZE, c=hf_fr_c, alpha=1.0,
           edgecolors="white", linewidths=0.8, zorder=5)

# HF labels — arcprize.org style: short names next to dots, thin leader lines for clusters
# Format: test_id -> (x_pos, y_pos, ha, label, leader)  — leader=True draws a thin line to the dot
hf_label_abs = {
    # Pareto frontier (>10% accuracy)
    "gemini-3-flash-preview-thinking-high":(0.40, 34.0, "left", "Gemini 3 Flash (high)", False),
    "grok-4.20-beta-0309b-reasoning":      (1.12, 63.6, "left", "Grok 4.20 (reasoning)", False),
    "gemini-3-1-pro-preview":              (1.12, 88.1, "left", "Gemini 3.1 Pro", False),
    # Opus 4.6 cluster — stacked labels to the right of dots
    "claude-opus-4-6-thinking-120K-high":  (4.00, 79.0, "left", "Opus 4.6 (120K, high)", False),
    "claude-opus-4-6-thinking-120K-max":   (4.00, 76.3, "left", "Opus 4.6 (120K, max)", False),
    "claude-opus-4-6-thinking-120K-medium":(3.22, 73.6, "left", "Opus 4.6 (120K, med)", False),
    # Lower-Q cluster at ($2, ~60) — spread labels via leader lines to clear DE Flash
    "claude-opus-4-6-thinking-120K-low":   (4.20, 66.5, "left", "Opus 4.6 (120K, low)", True),
    "gpt-5-2-2025-12-11-thinking-xhigh":   (4.20, 54.0, "left", "GPT-5.2 thinking (xhigh)", True),
    "claude-opus-4-5-20251101-thinking-64k":(2.70, 40.3, "left", "Opus 4.5 thinking (64K)", False),
    "gpt-5-2-pro-2025-12-11-medium":       (9.72, 37.9, "left", "GPT-5.2 Pro (med)", False),
    "gpt-5-pro-2025-10-06":                (8.20, 0.5, "left", "GPT-5 Pro", False),
    "gpt-5-1-2025-11-13-thinking-high":    (1.37, 18.8, "left", "GPT-5.1 (high)", False),
    "gemini-3-pro-preview":                (1.00, 28.3, "left", "Gemini 3 Pro (prev)", False),
}
all_pts = frontier_pts + [p for p in points if p not in frontier_pts]
labeled = set()
for p in all_pts:
    if p[3] != "hf":
        continue
    cost, score, tid, _ = p
    if tid in labeled or tid not in hf_label_abs:
        continue
    labeled.add(tid)
    tx, ty, ha, label, leader = hf_label_abs[tid]
    if leader:
        angleA = 180 if ha == "left" else 0
        angleB = 90 if ty > score else -90
        arrow = dict(arrowstyle="-", color="#999", alpha=0.55, lw=0.7,
                     connectionstyle=f"angle,angleA={angleA},angleB={angleB},rad=0",
                     shrinkA=3, shrinkB=4)
    else:
        arrow = None
    ax.annotate(label, xy=(cost, score),
                xytext=(tx, ty),
                fontsize=10, color=TEXT, ha=ha, zorder=6,
                arrowprops=arrow)

# Agentic + Athanor dots — arcprize.org style: name only (Athanor keeps detailed callout)
agentic_offsets_abs = {
    # name -> (x_pos, y_pos, ha)
    # All frontier/near-frontier labels placed ABOVE dots to clear the Pareto line
    "Confluence":                 (11.77, 99.6, "center"),   # above dot, clear of line
    "Squeeze-Evolve":             ( 5.93, 99.2, "center"),   # above dot, clear of line
    "Imbue DE":                   ( 8.90, 95.5, "left"),     # right of dot, between threshold (95) and Pareto line (~97.7)
    "Imbue DE (Flash)":           ( 2.65, 61.4, "left"),     # right of dot (no frontier line here)
    "Athanor (this work)":        ( 3.55, 87.5, "left"),     # below-right of triangle, clear of Pareto line
}
for name, cost, score, is_a in agentic:
    if is_a:
        # Soft concentric mint glow behind Athanor — draws the eye at thumbnail sizes
        for rad, alpha in [(3200, 0.07), (1700, 0.11), (900, 0.18)]:
            ax.scatter(cost, score, s=rad, c=ACCENT, marker="o",
                       alpha=alpha, edgecolors="none", zorder=7)
    ax.scatter(cost, score,
               s=ATHANOR_SIZE if is_a else DOT_SIZE,
               c=ACCENT if is_a else AGENT,
               marker="^" if is_a else "o",
               edgecolors="none",
               zorder=9 if is_a else 8)
    tx, ty, ha = agentic_offsets_abs[name]
    if is_a:
        label = f"{name}\n{score:.1f}%  •  ${cost:.2f}/task"
    else:
        label = name
    ax.annotate(label, xy=(cost, score),
                xytext=(tx, ty),
                fontsize=10,
                color=TEXT,
                ha=ha, zorder=10)

# Axes
ax.set_xlim(-0.35, 14)
ax.set_ylim(-2, 102)

ax.set_xlabel("Cost per Task  ($)", color=TEXT, fontsize=12, labelpad=6)
ax.set_ylabel("Public-Eval Score  (%)", color=TEXT, fontsize=12, labelpad=10)
ax.set_title("ARC-AGI-2 Public Eval — Cost vs. Accuracy",
             color=TEXT, fontsize=18, pad=16, fontweight="bold")

ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
ax.set_xticklabels(["$0", "$2", "$4", "$6", "$8", "$10", "$12", "$14"])
ax.set_yticks([0, 20, 40, 60, 80, 95, 100])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "95%", "100%"])

ax.tick_params(colors=TEXT, labelsize=10)
ax.grid(True, which="major", alpha=0.10, color="white", linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color("#333")
    spine.set_linewidth(0.8)

# "95% threshold" label placed on left inside band
ax.text(0.3, 95.5, "95% threshold", color=MUTED, fontsize=8.5,
        style="italic", ha="left", zorder=2)

# Legend
legend_items = [
    Line2D([0],[0], marker="^", color="none", markerfacecolor=ACCENT,
           markersize=14, label="Athanor (this work)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=AGENT,
           markersize=10, label="Other agentic systems"),
    Line2D([0],[0], color=FRONT, alpha=0.55, linestyle="--", linewidth=1.8,
           label="Pareto frontier"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=V_ANTHROPIC,
           markersize=9, label="Anthropic (Claude)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=V_OPENAI,
           markersize=9, label="OpenAI (GPT / o3)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=V_GEMINI,
           markersize=9, label="Google (Gemini)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=V_GROK,
           markersize=9, label="xAI (Grok)"),
]
leg = ax.legend(handles=legend_items, loc="lower right", frameon=False,
                fontsize=9.5, labelcolor=TEXT)

# Caption — anchored at figure bottom with va="bottom" so it renders below xlabel
fig.text(0.5, 0.010,
         "Snapshot as of April 19, 2026  •  "
         "HF corpus: arcprize/arc_agi_v2_public_eval (CoT-only, task-level)  •  "
         "Public eval only — not comparable to semi-private eval.",
         ha="center", va="bottom", fontsize=7.5, color=MUTED, style="italic")

plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.98])

# Draw "{" brace grouping the 4 CoT-only vendor entries, with "CoT-only" label to the left.
# Must run AFTER tight_layout so legend positions are final.
fig.canvas.draw()
vendor_texts = leg.get_texts()[-4:]  # Anthropic, OpenAI, Google, xAI
inv = fig.transFigure.inverted()
bboxes = [t.get_window_extent() for t in vendor_texts]
top_y = max(inv.transform((b.x0, b.y1))[1] for b in bboxes)
bot_y = min(inv.transform((b.x0, b.y0))[1] for b in bboxes)
left_x = min(inv.transform((b.x0, b.y0))[0] for b in bboxes)

brace_x = left_x - 0.030
mid_y  = (top_y + bot_y) / 2
q = 0.006  # notch depth (left)
s = 0.005  # top/bottom serifs (right)
brace_xs = [brace_x + s, brace_x, brace_x, brace_x - q, brace_x, brace_x, brace_x + s]
brace_ys = [top_y, top_y, mid_y + q*0.4, mid_y, mid_y - q*0.4, bot_y, bot_y]
fig.add_artist(Line2D(brace_xs, brace_ys, color=MUTED, lw=1.3,
                      transform=fig.transFigure,
                      solid_capstyle="round", solid_joinstyle="round"))
fig.text(brace_x - q - 0.004, mid_y, "CoT-only",
         ha="right", va="center", color=TEXT, fontsize=9.5, style="italic")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, facecolor=BG, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# Print frontier for sanity
print("\nPareto frontier points (sorted by cost):")
for p in frontier_pts:
    print(f"  ${p[0]:>6.2f}  {p[1]:>5.1f}%  {p[2][:40]}  ({p[3]})")
