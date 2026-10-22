"""HTML building blocks and CSS for the Streamlit app."""

from __future__ import annotations

from html import escape

from .formatting import fmt_compact, fmt_pct
from .reasons import AssetRecommendation, Reason

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #0f172a; --ink-2: #475569; --muted: #94a3b8;
  --line: #e5e8ef; --surface: #ffffff; --page: #f5f7fb;
  --blue: #2a78d6; --blue-soft: #e8f1fc; --orange: #eb6834;
  --good: #15803d; --good-soft: #e9f7ee; --bad: #b91c1c; --bad-soft: #fdecec;
  --warn: #b45309; --warn-soft: #fef4e2; --neutral-soft: #f1f5f9;
  --radius: 14px;
  --shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.05);
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
.block-container { padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1320px; }
[data-testid="stHeader"] { background: rgba(245,247,251,.85); backdrop-filter: blur(6px); }
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
@media (min-width: 900px) {
  section[data-testid="stSidebar"][aria-expanded="true"] { width: 440px !important; min-width: 440px !important; }
}
@media (max-width: 640px) {
  .hero { padding: 20px; } .hero-title { font-size: 24px; }
  .kpis { grid-template-columns: 1fr 1fr; gap: 8px; } .kpi-value { font-size: 18px; }
  .block-container { padding-left: 16px; padding-right: 16px; }
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* Hero */
.hero { background: linear-gradient(135deg, #0b1324 0%, #13284f 55%, #1d4f91 100%);
  border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 18px;
  position: relative; overflow: hidden; }
.hero:after { content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px;
  border-radius: 50%; background: radial-gradient(circle, rgba(57,135,229,.35), rgba(57,135,229,0) 70%); }
.hero-eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #9ec5f4; }
.hero-title { font-size: 30px; font-weight: 800; margin: 4px 0 6px; letter-spacing: -0.03em; color: #fff; }
.hero-sub { color: #c7d6ee; font-size: 14px; max-width: 760px; }
.pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.pill { font-size: 12px; font-weight: 500; padding: 4px 10px; border-radius: 999px;
  background: rgba(255,255,255,.10); color: #e2ebf8; border: 1px solid rgba(255,255,255,.14); }
.pill.warn { background: rgba(250,178,25,.18); border-color: rgba(250,178,25,.45); color: #ffe2a3; }

/* KPI tiles */
.kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
@media (max-width: 1100px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 14px 16px; box-shadow: var(--shadow); }
.kpi-label { font-size: 12px; color: var(--ink-2); font-weight: 500; display: flex; justify-content: space-between; }
.kpi-value { font-size: clamp(17px, 1.45vw, 22px); font-weight: 700; color: var(--ink); margin-top: 4px; letter-spacing: -0.02em; white-space: nowrap; }
.kpi-value .arrow { color: var(--muted); font-weight: 500; font-size: 15px; margin: 0 3px; }
.kpi-value .from { color: var(--muted); font-weight: 600; font-size: 15px; }
.kpi-sub { font-size: 12px; margin-top: 2px; color: var(--ink-2); }
.up { color: var(--good); font-weight: 600; } .down { color: var(--bad); font-weight: 600; }

/* Cards */
.card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 18px 20px; box-shadow: var(--shadow); margin-bottom: 14px; }
.card h3 { font-size: 16px; font-weight: 700; margin: 0 0 4px; color: var(--ink); }
.card .lede { color: var(--ink-2); font-size: 13px; margin-bottom: 10px; }
.section-title { font-size: 15px; font-weight: 700; color: var(--ink); margin: 6px 0 2px; }
.section-sub { font-size: 13px; color: var(--ink-2); margin-bottom: 8px; }

/* Verdict */
.verdict { display: grid; grid-template-columns: 1.15fr 1fr; gap: 28px; }
@media (max-width: 900px) { .verdict { grid-template-columns: 1fr; } }
.verdict-head { font-size: 20px; font-weight: 750; color: var(--ink); line-height: 1.3; letter-spacing: -0.02em; }
.verdict-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }

/* Reasons */
ul.reasons { list-style: none; padding: 0; margin: 0; }
ul.reasons li { display: grid; grid-template-columns: 22px 1fr; gap: 8px; padding: 7px 0;
  border-top: 1px solid #f0f2f6; font-size: 13.5px; color: var(--ink); line-height: 1.45; margin: 0; }
ul.reasons li:first-child { border-top: none; }
.ricon { width: 20px; height: 20px; border-radius: 6px; display: inline-flex; align-items: center;
  justify-content: center; font-size: 12px; font-weight: 700; margin-top: 1px; }
.ricon.positive { background: var(--good-soft); color: var(--good); }
.ricon.negative { background: var(--bad-soft); color: var(--bad); }
.ricon.neutral { background: var(--neutral-soft); color: var(--ink-2); }
.rcat { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em;
  color: var(--muted); margin-right: 6px; }

/* Badges & chips */
.badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 700;
  padding: 4px 10px; border-radius: 999px; white-space: nowrap; }
.badge:before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.b-buy { background: var(--good-soft); color: var(--good); }
.b-trim { background: var(--warn-soft); color: var(--warn); }
.b-exit { background: var(--bad-soft); color: var(--bad); }
.b-hold { background: var(--neutral-soft); color: var(--ink-2); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px; }
.chip { font-size: 12px; color: var(--ink-2); background: #f6f8fb; border: 1px solid var(--line);
  border-radius: 8px; padding: 3px 8px; }
.chip b { color: var(--ink); font-weight: 600; }

/* Recommendation card */
.rec { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 18px 20px; box-shadow: var(--shadow); margin-bottom: 14px; }
.rec-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.rec-ticker { font-size: 18px; font-weight: 750; color: var(--ink); letter-spacing: -0.01em; }
.rec-name { font-size: 12.5px; color: var(--ink-2); }
.rec-headline { font-size: 14px; font-weight: 600; color: var(--ink); margin: 12px 0 8px; }
.wbar { display: grid; grid-template-columns: 58px 1fr 48px; align-items: center; gap: 8px;
  font-size: 12px; color: var(--ink-2); margin: 4px 0; }
.wtrack { height: 8px; background: #f1f4f8; border-radius: 99px; overflow: hidden; }
.wfill { height: 100%; border-radius: 99px; }
.wval { text-align: right; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.rgroup { font-size: 12px; font-weight: 700; color: var(--ink-2); margin: 14px 0 2px;
  text-transform: uppercase; letter-spacing: .06em; }
details.more summary { cursor: pointer; font-size: 12px; font-weight: 700; color: var(--ink-2);
  text-transform: uppercase; letter-spacing: .06em; margin-top: 14px; list-style: none; }
details.more summary::-webkit-details-marker { display: none; }
details.more summary:after { content: " ▾"; color: var(--muted); }
details.more[open] summary:after { content: " ▴"; }

.cashline { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 13px; color: var(--ink-2); }
.cashline .chip { margin: 0; }
.cashnote b { color: var(--ink); }
.chip.conf-High { background: var(--good-soft); border-color: #cdebd7; }
.chip.conf-Medium { background: var(--warn-soft); border-color: #f6e2bd; }
.chip.conf-Low { background: var(--bad-soft); border-color: #f5cfcf; }
.tldr { background: var(--blue-soft); border-radius: 10px; padding: 10px 12px; margin: 10px 0 4px;
  font-size: 14px; color: var(--ink); line-height: 1.45; }
.tldr b { color: #1c5cab; }
ol.steps { margin: 12px 0 0; padding: 0; list-style: none; counter-reset: step; }
ol.steps li { counter-increment: step; display: grid; grid-template-columns: 26px 1fr; gap: 10px;
  font-size: 14px; color: var(--ink); line-height: 1.45; margin: 0 0 10px; }
ol.steps li:before { content: counter(step); width: 24px; height: 24px; border-radius: 50%; background: var(--blue);
  color: #fff; font-weight: 700; font-size: 12px; display: inline-flex; align-items: center; justify-content: center; }
details.why summary { cursor: pointer; font-size: 12px; font-weight: 700; color: var(--ink-2);
  text-transform: uppercase; letter-spacing: .06em; margin-top: 12px; list-style: none; }
details.why summary::-webkit-details-marker { display: none; }
details.why summary:after { content: " ▾"; color: var(--muted); }
details.why[open] summary:after { content: " ▴"; }
.score { display: inline-block; min-width: 34px; text-align: center; font-weight: 700; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 10px 14px; font-weight: 600; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background: var(--surface); border-radius: 12px; }
.footnote { font-size: 12px; color: var(--muted); margin-top: 28px; text-align: center; }
</style>
"""

_ICONS = {"positive": "✓", "negative": "✕", "neutral": "•"}
_BADGE = {"Buy more": "b-buy", "Initiate": "b-buy", "Trim": "b-trim", "Exit": "b-exit", "Hold": "b-hold"}


def badge(action: str, subject: str = "") -> str:
    text = f"{action} {subject}".strip()
    return f'<span class="badge {_BADGE.get(action, "b-hold")}">{escape(text)}</span>'


def reason_list(reasons: list[Reason], show_category: bool = True) -> str:
    items = []
    for r in reasons:
        cat = f'<span class="rcat">{escape(r.category)}</span>' if show_category else ""
        items.append(f'<li><span class="ricon {r.kind}">{_ICONS[r.kind]}</span>'
                     f'<span>{cat}{escape(r.text)}</span></li>')
    return f'<ul class="reasons">{"".join(items)}</ul>'


def hero(title: str, subtitle: str, pills: list[tuple[str, str]]) -> str:
    pills_html = "".join(f'<span class="pill {cls}">{escape(text)}</span>' for text, cls in pills)
    return (f'<div class="hero"><div class="hero-eyebrow">Portfolio optimizer</div>'
            f'<div class="hero-title">{escape(title)}</div><div class="hero-sub">{escape(subtitle)}</div>'
            f'<div class="pills">{pills_html}</div></div>')


def kpi(label: str, value_html: str, sub_html: str = "", hint: str = "") -> str:
    hint_html = f'<span title="{escape(hint)}" style="cursor:help;color:#94a3b8">ⓘ</span>' if hint else ""
    return (f'<div class="kpi"><div class="kpi-label"><span>{escape(label)}</span>{hint_html}</div>'
            f'<div class="kpi-value">{value_html}</div><div class="kpi-sub">{sub_html}</div></div>')


def change(before: str, after: str) -> str:
    return f'<span class="from">{escape(before)}</span><span class="arrow">→</span>{escape(after)}'


def delta(x: float, fmt, good_if_up: bool = True) -> str:
    if abs(x) < 1e-9:
        return '<span>no change</span>'
    good = (x > 0) == good_if_up
    return f'<span class="{"up" if good else "down"}">{"▲" if x > 0 else "▼"} {escape(fmt(abs(x)))}</span>'


def _wbar(label: str, value: float, scale: float, color: str) -> str:
    width = 0 if scale <= 0 else min(value / scale, 1) * 100
    return (f'<div class="wbar"><span>{label}</span><div class="wtrack">'
            f'<div class="wfill" style="width:{width:.1f}%;background:{color}"></div></div>'
            f'<span class="wval">{value:.1%}</span></div>')


def rec_card(rec: AssetRecommendation, currency: str, scale: float) -> str:
    snap = rec.technical
    score_cls = "up" if snap.score >= 25 else "down" if snap.score <= -25 else ""
    pnl_cls = "up" if rec.pnl >= 0 else "down"
    chips = [
        f'<span class="chip">Price <b>{currency}{rec.price:,.2f}</b></span>',
        f'<span class="chip">Timing <b>{escape(rec.timing)}</b></span>',
        f'<span class="chip">Technicals <b class="{score_cls}">{snap.stance} {snap.score:+.0f}</b></span>',
        f'<span class="chip">Exp. return <b>{fmt_pct(rec.expected_return)}</b></span>',
        f'<span class="chip">Volatility <b>{fmt_pct(rec.volatility)}</b></span>',
    ]
    if rec.confidence and rec.weight_range:
        lo, hi = (f"{x:.0%}" for x in rec.weight_range)
        span = lo if lo == hi else f"{lo}–{hi}"
        chips.insert(2, f'<span class="chip conf-{rec.confidence}" title="10–90% range of the optimal weight '
                        f'across resampled price histories">Confidence <b>{rec.confidence}</b> · {span}</span>')
    if rec.qty > 0 and rec.avg_price > 0:
        chips.append(f'<span class="chip">P&amp;L <b class="{pnl_cls}">{fmt_pct(rec.pnl_pct, 1, signed=True)}'
                     f' ({fmt_compact(rec.pnl, currency, signed=True)})</b></span>')

    why = [r for r in rec.reasons if r.category not in ("Technical", "Position")]
    tech = [r for r in rec.reasons if r.category == "Technical"]
    pos = [r for r in rec.reasons if r.category == "Position"]
    tech_summary = sum(r.kind == "positive" for r in tech), sum(r.kind == "negative" for r in tech)

    body = (
        f'<details class="why"><summary>Why · {len(why)} reasons</summary>{reason_list(why)}</details>'
        + (f'<details class="more"><summary>Timing &amp; technicals · {tech_summary[0]} bullish, '
           f'{tech_summary[1]} bearish</summary>{reason_list(tech, show_category=False)}</details>' if tech else "")
        + (f'<div class="rgroup">Your position</div>{reason_list(pos, show_category=False)}' if pos else "")
    )
    return (
        f'<div class="rec"><div class="rec-head"><div><div class="rec-ticker">{escape(rec.ticker)}</div>'
        f'<div class="rec-name">{escape(rec.label)} · {escape(rec.asset_class)}</div></div>{badge(rec.action)}</div>'
        f'<div class="rec-headline">{escape(rec.headline)}</div>'
        + (f'<div class="tldr"><b>In short.</b> {escape(rec.summary)}</div>' if rec.summary else "")
        + f'{_wbar("Now", rec.current_weight, scale, "#eb6834")}{_wbar("Target", rec.target_weight, scale, "#2a78d6")}'
        f'<div class="chips">{"".join(chips)}</div>{body}</div>'
    )


def plan_steps(steps: list[str]) -> str:
    return '<ol class="steps">' + "".join(f"<li><span>{escape(s)}</span></li>" for s in steps) + "</ol>"
