#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: pcb_exec_summary_ui.py
Python 3.7+ (local-first)
Deps: pandas (numpy optional)

Genera una Executive Summary "UI" (HTML) leggibile da browser:
- out/executive_summary.html
- out/executive_summary.json

Input (best-effort, se mancano alcuni file li salta):
- out/insights_level2.csv
- out/edges.csv (se esiste)
- out/alerts_today_level28.csv
- out/experiment_summary_level29.csv
- out/experiment_results.csv
- out/insights_level3.csv
- out/insights_level3_ledger.csv
- out/experiment_trials_enriched.csv

Uso:
  python pcb_exec_summary_ui.py
  python pcb_exec_summary_ui.py --out out/executive_summary.html
  python pcb_exec_summary_ui.py --title "PCB Executive Summary"
"""

import os
import sys
import json
import argparse
from datetime import datetime

import pandas as pd

OUT_DIR = "out"

DEFAULT_OUT_HTML = os.path.join(OUT_DIR, "executive_summary.html")
DEFAULT_OUT_JSON = os.path.join(OUT_DIR, "executive_summary.json")

INSIGHTS_L2 = os.path.join(OUT_DIR, "insights_level2.csv")
EDGES = os.path.join(OUT_DIR, "edges.csv")  # optional
ALERTS_L28 = os.path.join(OUT_DIR, "alerts_today_level28.csv")
EXP_SUMMARY_L29 = os.path.join(OUT_DIR, "experiment_summary_level29.csv")
EXP_RESULTS = os.path.join(OUT_DIR, "experiment_results.csv")
INSIGHTS_L3 = os.path.join(OUT_DIR, "insights_level3.csv")
LEDGER_L3 = os.path.join(OUT_DIR, "insights_level3_ledger.csv")
TRIALS_ENRICHED = os.path.join(OUT_DIR, "experiment_trials_enriched.csv")


def _exists(p):
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _read_csv_best_effort(path):
    if not _exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _safe_float(x, default=None):
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _fmt_pct(x):
    v = _safe_float(x, None)
    if v is None:
        return "—"
    return "{:.0f}%".format(100.0 * v)


def _trim(s, n=160):
    try:
        s = "" if s is None else str(s)
    except Exception:
        s = ""
    s = s.replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _pick_col(df, candidates):
    if df is None or len(df) == 0:
        return None
    cols = set([c for c in df.columns])
    for c in candidates:
        if c in cols:
            return c
    return None


def _df_to_html_table(df, max_rows=20):
    if df is None or len(df) == 0:
        return "<div class='muted'>Nessun dato disponibile.</div>"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    return d.to_html(index=False, escape=True, classes="tbl")


def _score_health(files_present, warnings):
    present = sum(1 for _, ok in files_present.items() if ok)
    total = len(files_present)
    base = present / float(total) if total > 0 else 0.0
    penalty = min(0.25, 0.05 * len(warnings))
    return max(0.0, min(1.0, base - penalty))


def _status_badge(text, kind="neutral"):
    return "<span class='badge badge-{}'>{}</span>".format(kind, text)


def build_exec_summary(title="PCB — Executive Summary", out_html=DEFAULT_OUT_HTML, out_json=DEFAULT_OUT_JSON):
    _ensure_out_dir()

    df_l2 = _read_csv_best_effort(INSIGHTS_L2)
    df_edges = _read_csv_best_effort(EDGES)
    df_alerts = _read_csv_best_effort(ALERTS_L28)
    df_exp_sum = _read_csv_best_effort(EXP_SUMMARY_L29)
    df_exp = _read_csv_best_effort(EXP_RESULTS)
    df_l3 = _read_csv_best_effort(INSIGHTS_L3)
    df_led = _read_csv_best_effort(LEDGER_L3)
    df_trials = _read_csv_best_effort(TRIALS_ENRICHED)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    files_present = {
        "insights_level2.csv": df_l2 is not None,
        "edges.csv": df_edges is not None,
        "alerts_today_level28.csv": df_alerts is not None,
        "experiment_summary_level29.csv": df_exp_sum is not None,
        "experiment_results.csv": df_exp is not None,
        "insights_level3.csv": df_l3 is not None,
        "insights_level3_ledger.csv": df_led is not None,
        "experiment_trials_enriched.csv": df_trials is not None,
    }

    warnings = []
    n_l2 = int(len(df_l2)) if df_l2 is not None else 0
    n_edges = int(len(df_edges)) if df_edges is not None else 0
    n_alerts = int(len(df_alerts)) if df_alerts is not None else 0
    n_trials_logged = int(len(df_exp)) if df_exp is not None else 0

    top_actions = None
    if df_exp_sum is not None and len(df_exp_sum) > 0:
        sr_col = _pick_col(df_exp_sum, ["success_rate", "success_rate_lb"])
        z_col = _pick_col(df_exp_sum, ["avg_z", "avg_z_cf"])
        sort_cols = [c for c in [sr_col, z_col] if c]
        if sort_cols:
            top_actions = df_exp_sum.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()
        else:
            top_actions = df_exp_sum.copy()
        top_actions = top_actions.head(10)

    n_supported = 0
    supported = None
    if df_l3 is not None and len(df_l3) > 0:
        status_col = _pick_col(df_l3, ["status"])
        if status_col:
            supported = df_l3[df_l3[status_col].astype(str).isin(["action_supported", "supported", "robust"])].copy()
            n_supported = int(len(supported))
        else:
            supported = df_l3.copy()
            n_supported = int(len(supported))

        lb_col = _pick_col(df_l3, ["success_rate_lb", "success_rate"])
        if lb_col and supported is not None and len(supported) > 0:
            supported = supported.sort_values([lb_col], ascending=[False]).head(10)

    if df_l2 is None or len(df_l2) == 0:
        warnings.append("Nessun insight Level 2 trovato: esegui pipeline 2.6 → 2.5 prima di generare la summary.")
    if df_exp is None or len(df_exp) == 0:
        warnings.append("Nessun trial registrato: il valore della validazione (L3) aumenta molto dopo 5–15 trial reali.")
    if df_l3 is None or len(df_l3) == 0:
        warnings.append("Nessun output Level 3: esegui la validazione (3.2) dopo aver loggato trial.")
    if df_alerts is None:
        warnings.append("Alert Level 2.8 mancante (non critico, ma utile per UX).")

    health = _score_health(files_present, warnings)

    if health >= 0.75:
        health_badge = _status_badge("Pipeline OK", "good")
    elif health >= 0.45:
        health_badge = _status_badge("Parziale", "warn")
    else:
        health_badge = _status_badge("Incompleto", "bad")

    summary_json = {
        "generated_at": now,
        "title": title,
        "files_present": files_present,
        "metrics": {
            "insights_level2": n_l2,
            "edges": n_edges,
            "alerts_today": n_alerts,
            "trials_logged": n_trials_logged,
            "insights_level3_supported": n_supported,
            "health_score_0_1": float(health),
        },
        "warnings": warnings,
    }

    css = """
    :root { --bg:#0b0f14; --card:#111824; --muted:#8aa0b5; --text:#eaf2ff; --line:rgba(255,255,255,0.08);
            --good:#2ee59d; --warn:#ffd166; --bad:#ff5c77; --accent:#7cc4ff; }
    *{box-sizing:border-box;}
    body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
         background:radial-gradient(1200px 600px at 10% 0%, rgba(124,196,255,0.12), transparent),
                    radial-gradient(800px 400px at 90% 10%, rgba(46,229,157,0.10), transparent), var(--bg);
         color:var(--text);line-height:1.35;}
    .wrap{max-width:1120px;margin:0 auto;padding:28px 18px 60px;}
    .topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;}
    h1{margin:0;font-size:26px;letter-spacing:.2px;}
    .sub{color:var(--muted);margin-top:8px;font-size:13px;}
    .badge{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);
           padding:7px 10px;border-radius:999px;font-size:12px;background:rgba(255,255,255,0.03);}
    .badge-good{border-color:rgba(46,229,157,0.35);}
    .badge-warn{border-color:rgba(255,209,102,0.35);}
    .badge-bad{border-color:rgba(255,92,119,0.35);}
    .grid{margin-top:18px;display:grid;grid-template-columns:repeat(12,1fr);gap:14px;}
    .card{grid-column:span 12;background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
          border:1px solid var(--line);border-radius:18px;padding:16px 16px 14px;
          box-shadow:0 16px 40px rgba(0,0,0,0.35);}
    .card h2{margin:0 0 10px;font-size:16px;}
    .kpis{display:grid;grid-template-columns:repeat(12,1fr);gap:12px;}
    .kpi{grid-column:span 3;background:rgba(255,255,255,0.03);border:1px solid var(--line);
         border-radius:16px;padding:12px;}
    .kpi .label{color:var(--muted);font-size:12px;}
    .kpi .value{font-size:22px;margin-top:6px;}
    .kpi .hint{color:var(--muted);font-size:12px;margin-top:6px;}
    .muted{color:var(--muted);}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
    @media(max-width:980px){.kpi{grid-column:span 6;}.two{grid-template-columns:1fr;}}
    .tbl{width:100%;border-collapse:collapse;font-size:12.5px;border-radius:12px;overflow:hidden;}
    .tbl th,.tbl td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top;}
    .tbl th{color:var(--muted);font-weight:600;}
    .tbl tr:hover td{background:rgba(124,196,255,0.05);}
    .warnbox{background:rgba(255,209,102,0.08);border:1px solid rgba(255,209,102,0.25);
             border-radius:14px;padding:12px;margin-top:10px;font-size:13px;}
    .okbox{background:rgba(46,229,157,0.08);border:1px solid rgba(46,229,157,0.25);
           border-radius:14px;padding:12px;margin-top:10px;font-size:13px;}
    .footer{margin-top:22px;color:var(--muted);font-size:12px;}
    .pill{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted);font-size:12px;}
    """

    checklist_rows = [{"Artifact": os.path.join(OUT_DIR, k), "Status": ("OK" if ok else "MISSING")} for k, ok in files_present.items()]
    df_check = pd.DataFrame(checklist_rows)

    next_steps = []
    if n_trials_logged < 5:
        next_steps.append("Logga 5–10 trial reali (Level 2.9 log) per rendere la validazione L3 più credibile.")
    if df_l3 is None or len(df_l3) == 0:
        next_steps.append("Esegui la validazione (Level 3.2) e genera insights_level3.csv + ledger.")
    if df_edges is None:
        next_steps.append("Esporta edges.csv (opzionale) per una vista grafo più immediata.")
    if not next_steps:
        next_steps.append("Pipeline completa: puoi presentare l’output come demo esecutiva (partner/procurement).")

    highlights = []
    if n_supported > 0:
        highlights.append("Validazioni supportate (L3): {}".format(n_supported))
    if n_alerts > 0:
        highlights.append("Alert oggi (L28): {}".format(n_alerts))
    if n_l2 > 0:
        highlights.append("Insight scoperti (L2): {}".format(n_l2))
    if not highlights:
        highlights.append("Esegui la pipeline per generare output e riaprire questa pagina.")

    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'/>")
    html.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    html.append("<title>{}</title>".format(_trim(title, 80)))
    html.append("<style>{}</style>".format(css))
    html.append("</head><body><div class='wrap'>")
    html.append("<div class='topbar'><div>")
    html.append("<h1>{}</h1>".format(_trim(title, 140)))
    html.append("<div class='sub'>Generated: {} · Local-first · Audit-ready</div>".format(now))
    html.append("</div><div>{}</div></div>".format(health_badge))

    html.append("<div class='grid'>")
    html.append("<div class='card'><h2>Executive Snapshot</h2>")
    html.append("<div class='kpis'>")
    html.append("<div class='kpi'><div class='label'>Health score</div><div class='value'>{}</div><div class='hint'>Copertura artifact + warnings</div></div>".format(_fmt_pct(health)))
    html.append("<div class='kpi'><div class='label'>Insight Level 2</div><div class='value'>{}</div><div class='hint'>Discovery</div></div>".format(n_l2))
    html.append("<div class='kpi'><div class='label'>Trials loggati</div><div class='value'>{}</div><div class='hint'>Esperimenti (2.9)</div></div>".format(n_trials_logged))
    html.append("<div class='kpi'><div class='label'>Supportati (L3)</div><div class='value'>{}</div><div class='hint'>Validazione controfattuale</div></div>".format(n_supported))
    html.append("</div>")

    html.append("<div style='margin-top:12px'>")
    for h in highlights:
        html.append("<span class='pill' style='margin-right:8px'>{}</span>".format(_trim(h, 80)))
    html.append("</div>")

    if warnings:
        html.append("<div class='warnbox'><b>Attenzioni</b><ul style='margin:8px 0 0 18px'>")
        for w in warnings:
            html.append("<li>{}</li>".format(_trim(w, 220)))
        html.append("</ul></div>")
    else:
        html.append("<div class='okbox'><b>OK</b> — Nessuna attenzione critica rilevata.</div>")
    html.append("</div>")

    html.append("<div class='card'><h2>Top Findings</h2><div class='two'>")

    html.append("<div><div class='muted' style='margin-bottom:8px'>Validazione (Level 3) — top</div>")
    if supported is not None and len(supported) > 0:
        cols = [c for c in ["insight_id","source","lag","n_trials","n_trials_scored","success_rate_lb","avg_z_cf","status"] if c in supported.columns]
        if not cols:
            cols = list(supported.columns)[:8]
        html.append(_df_to_html_table(supported[cols], max_rows=10))
    else:
        html.append("<div class='muted'>Nessuna validazione supportata trovata (o file mancante).</div>")
    html.append("</div>")

    html.append("<div><div class='muted' style='margin-bottom:8px'>Esperimenti (Level 2.9) — top actions</div>")
    if top_actions is not None and len(top_actions) > 0:
        cols = [c for c in ["insight_id","action_name","n_trials","success_rate","avg_z","median_z"] if c in top_actions.columns]
        if not cols:
            cols = list(top_actions.columns)[:8]
        html.append(_df_to_html_table(top_actions[cols], max_rows=10))
    else:
        html.append("<div class='muted'>Nessun summary esperimenti trovato (o file mancante).</div>")
    html.append("</div>")

    html.append("</div></div>")

    html.append("<div class='card'><h2>Today Alerts (Level 2.8)</h2>")
    if df_alerts is not None and len(df_alerts) > 0:
        cols = list(df_alerts.columns)
        preferred = [c for c in ["priority","source","z","z_score","action","reason","notes","date"] if c in cols]
        if not preferred:
            preferred = cols[:10]
        html.append(_df_to_html_table(df_alerts[preferred], max_rows=20))
    else:
        html.append("<div class='muted'>Nessun alert disponibile (o file mancante).</div>")
    html.append("</div>")

    html.append("<div class='card'><h2>Artifact Checklist (for partner / procurement)</h2>")
    html.append("<div class='muted' style='margin-bottom:8px'>Dimostra che l’output contract è rispettato.</div>")
    html.append(_df_to_html_table(df_check, max_rows=50))
    html.append("</div>")

    html.append("<div class='card'><h2>Recommended Next Steps</h2><ol style='margin:8px 0 0 18px'>")
    for s in next_steps:
        html.append("<li>{}</li>".format(_trim(s, 240)))
    html.append("</ol><div class='footer'>Generated by pcb_exec_summary_ui.py · Local-first · No external calls</div></div>")

    html.append("</div></div></body></html>")

    html_str = "\n".join(html)

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_str)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    return out_html, out_json


def build_parser():
    p = argparse.ArgumentParser(
        prog="pcb_exec_summary_ui.py",
        description="Generate PCB Executive Summary UI (single-file HTML) from out/*.csv artifacts.",
    )
    p.add_argument("--out", default=DEFAULT_OUT_HTML, help="Output HTML path (default: out/executive_summary.html)")
    p.add_argument("--json", default=DEFAULT_OUT_JSON, help="Output JSON path (default: out/executive_summary.json)")
    p.add_argument("--title", default="PCB — Executive Summary", help="Page title")
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    out_html, out_json = build_exec_summary(title=args.title, out_html=args.out, out_json=args.json)
    print("\n[pcb] Executive Summary generated:")
    print(" - HTML:", out_html)
    print(" - JSON:", out_json)
    print("\nApri il file HTML nel browser per vedere la UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
