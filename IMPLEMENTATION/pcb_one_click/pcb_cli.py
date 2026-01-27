# FILE: pcb_cli.py
# Python 3.7 compatible
#
# PCB — Unified CLI ("pcb")
#
# Orchestrates: 2.6 -> 2.5 -> 2.8 -> 2.9 -> 3.2
#
# Usage:
#   python src/pcb_cli.py run
#   python -m pcb_cli run               (after install)
#   pcb run                             (after install via entrypoint)
#
# Dependencies: stdlib + numpy + pandas (required by underlying levels)

import os
import sys
import json
import argparse
from pcb_graph_export import export_personal_causal_graph
from pcb_validate_pcg import validate_pcg_file
import traceback
import traceback

# Ensure we can import sibling modules when running from repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

OUT_DIR = "out"
DEFAULT_DATA_CSV = "data.csv"
FALLBACK_DATA_CSV = os.path.join(OUT_DIR, "demo_data.csv")

# Canonical outputs (contract)
OUT_DATA_CLEAN = os.path.join(OUT_DIR, "data_clean.csv")
OUT_INSIGHTS_L2 = os.path.join(OUT_DIR, "insights_level2.csv")
OUT_EDGES = os.path.join(OUT_DIR, "edges.csv")
OUT_ALERTS_L28 = os.path.join(OUT_DIR, "alerts_today_level28.csv")
OUT_EXP_PLAN = os.path.join(OUT_DIR, "experiment_plan.csv")
OUT_EXP_RESULTS = os.path.join(OUT_DIR, "experiment_results.csv")
OUT_EXP_SUMMARY = os.path.join(OUT_DIR, "experiment_summary_level29.csv")
OUT_INSIGHTS_L3 = os.path.join(OUT_DIR, "insights_level3.csv")
OUT_LEDGER_L3 = os.path.join(OUT_DIR, "insights_level3_ledger.csv")
# current Level 3.2 file name in this package
OUT_TRIALS_ENRICHED = os.path.join(OUT_DIR, "experiment_trials_enriched_level32.csv")

DEFAULT_CONFIG_JSON = "pcb.json"


def _ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def _exists(p):
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def _load_config(config_path):
    """Minimal config loader (JSON only). If missing, returns defaults."""
    cfg = {
        "target": "mood",
        "use_data_clean_if_present": True,
        "pipeline": {
            "run_level26": True,
            "run_level25": True,
            "run_level28": True,
            "run_level29_plan": True,
            "run_level29_eval": True,
            "run_level32": True,
        },
        "level25": {"max_lag": 7},
    }

    if not config_path or not _exists(config_path):
        return cfg

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f) or {}
        for k, v in user_cfg.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except Exception:
        print("[pcb] WARNING: could not parse config, using defaults:", config_path)

    return cfg


def _pick_data_path(preferred=None, allow_clean=True):
    """Choose input data path with sane defaults."""
    if preferred and _exists(preferred):
        return preferred
    if allow_clean and _exists(OUT_DATA_CLEAN):
        return OUT_DATA_CLEAN
    if _exists(DEFAULT_DATA_CSV):
        return DEFAULT_DATA_CSV
    return FALLBACK_DATA_CSV


def _call_main(module_name, main_kwargs=None, argv=None):
    """
    Import module and call its main() robustly.
      - If argv is provided and module has main(argv), call that.
      - Else call main(**main_kwargs) if possible.
      - Else call main().
    """
    main_kwargs = main_kwargs or {}
    try:
        mod = __import__(module_name, fromlist=["main"])
    except Exception as e:
        print("[pcb] ERROR: cannot import module:", module_name)
        print("       ", str(e))
        return False

    if not hasattr(mod, "main"):
        print("[pcb] ERROR: module has no main():", module_name)
        return False

    fn = getattr(mod, "main")

    if argv is not None:
        try:
            fn(argv)
            return True
        except TypeError:
            pass
        except SystemExit:
            return True
        except Exception:
            print("[pcb] ERROR while running:", module_name, "main(argv)")
            traceback.print_exc()
            return False

    try:
        fn(**main_kwargs)
        return True
    except TypeError:
        try:
            fn()
            return True
        except Exception:
            print("[pcb] ERROR while running:", module_name, "main()")
            traceback.print_exc()
            return False
    except Exception:
        print("[pcb] ERROR while running:", module_name, "main(**kwargs)")
        traceback.print_exc()
        return False


def _print_outputs_summary():
    print("\n[pcb] Output contract (check these files):")
    for p in [
        OUT_DATA_CLEAN,
        OUT_EDGES,
        OUT_INSIGHTS_L2,
        OUT_ALERTS_L28,
        OUT_EXP_PLAN,
        OUT_EXP_RESULTS,
        OUT_EXP_SUMMARY,
        OUT_INSIGHTS_L3,
        OUT_LEDGER_L3,
        OUT_TRIALS_ENRICHED,
    ]:
        status = "OK" if _exists(p) else "MISSING"
        print(" - %-44s %s" % (p, status))


def cmd_init(args):
    _ensure_out()

    readme = os.path.join(OUT_DIR, "README_OUT.txt")
    if not _exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "This folder is generated by PCB.\n\n"
                "Canonical outputs:\n"
                "- out/data_clean.csv\n"
                "- out/edges.csv\n"
                "- out/insights_level2.csv\n"
                "- out/alerts_today_level28.csv\n"
                "- out/experiment_plan.csv\n"
                "- out/experiment_results.csv\n"
                "- out/experiment_summary_level29.csv\n"
                "- out/insights_level3.csv\n"
                "- out/insights_level3_ledger.csv\n"
                "- out/experiment_trials_enriched_level32.csv\n"
            )

    if args.scaffold_trials and not _exists(OUT_EXP_RESULTS):
        with open(OUT_EXP_RESULTS, "w", encoding="utf-8") as f:
            f.write("insight_id,action_name,date,t_index,adherence_flag,dose,notes\n")

    print("[pcb] init done. out/ is ready.")
    _print_outputs_summary()
    return 0


def cmd_run(args):
    _ensure_out()

    cfg = _load_config(args.config)
    allow_clean = bool(cfg.get("use_data_clean_if_present", True))

    data_in = _pick_data_path(preferred=args.data, allow_clean=allow_clean)
    if not _exists(data_in):
        print("[pcb] ERROR: data file not found. Provide --data or create data.csv.")
        return 2

    print("[pcb] Using data:", data_in)

    # 2.6
    if cfg.get("pipeline", {}).get("run_level26", True) and not args.skip_26:
        ok = _call_main("pcb_data_level26", main_kwargs={"data_csv_path": data_in})
        if not ok:
            print("[pcb] Level 2.6 failed.")
            return 2
        if _exists(OUT_DATA_CLEAN):
            data_in = OUT_DATA_CLEAN
            print("[pcb] Switched to cleaned data:", data_in)

    # 2.5
    if cfg.get("pipeline", {}).get("run_level25", True) and not args.skip_25:
        target = cfg.get("target", "mood")
        max_lag = int(cfg.get("level25", {}).get("max_lag", 7))
        ok = _call_main(
            "pcb_insights_level25",
            main_kwargs={"data_csv_path": data_in, "target_col": target, "max_lag": max_lag},
        )
        if not ok:
            print("[pcb] Level 2.5 failed.")
            return 2

    
    # Export Personal Causal Graph (required)
    print("[pcb] Exporting Personal Causal Graph artifact (out/personal_causal_graph.json)...")
    try:
        pcg_path = export_personal_causal_graph(out_dir=cfg.get("out_dir", "out"))
        ok, errs = validate_pcg_file(pcg_path)
        if not ok:
            raise RuntimeError("PCG validation failed: " + "; ".join(errs))
    except Exception:
        print("[pcb] ERROR: failed to export/validate Personal Causal Graph.", file=sys.stderr)
        traceback.print_exc()
        return 2

# 2.8
    if cfg.get("pipeline", {}).get("run_level28", True) and not args.skip_28:
        ok = _call_main(
            "pcb_alerts_level28",
            main_kwargs={"data_csv_path": data_in, "insights_path": OUT_INSIGHTS_L2},
        )
        if not ok:
            print("[pcb] Level 2.8 failed.")
            return 2

    # 2.9 plan
    if cfg.get("pipeline", {}).get("run_level29_plan", True) and not args.skip_29_plan:
        ok = _call_main("pcb_experiments_level29", argv=["plan"])
        if not ok:
            print("[pcb] Level 2.9 plan failed.")
            return 2

    # 2.9 eval
    if cfg.get("pipeline", {}).get("run_level29_eval", True) and not args.skip_29_eval:
        ok = _call_main("pcb_experiments_level29", argv=["eval", "--data", data_in])
        if not ok:
            print("[pcb] Level 2.9 eval failed.")
            return 2

    # 3.2
    if cfg.get("pipeline", {}).get("run_level32", True) and not args.skip_32:
        ok = _call_main("pcb_level3_engine_32", main_kwargs={})
        if not ok:
            print("[pcb] Level 3.2 failed.")
            return 2

    print("\n[pcb] run complete.")
    _print_outputs_summary()
    return 0


def cmd_plan(args):
    _ensure_out()
    ok = _call_main("pcb_experiments_level29", argv=["plan"])
    return 0 if ok else 2


def cmd_log(args):
    _ensure_out()
    argv = [
        "log",
        "--insight_id", args.insight_id,
        "--adherence_flag", str(int(args.adherence_flag)),
        "--notes", args.notes or "",
    ]
    if args.action_name:
        argv += ["--action_name", args.action_name]
    if args.date:
        argv += ["--date", args.date]
    if args.t_index is not None:
        argv += ["--t_index", str(int(args.t_index))]
    if args.dose:
        argv += ["--dose", args.dose]
    if args.data:
        argv += ["--data", args.data]

    ok = _call_main("pcb_experiments_level29", argv=argv)
    return 0 if ok else 2


def cmd_eval(args):
    _ensure_out()
    data_in = _pick_data_path(preferred=args.data, allow_clean=True)
    ok = _call_main("pcb_experiments_level29", argv=["eval", "--data", data_in])
    return 0 if ok else 2


def cmd_validate(args):
    _ensure_out()
    ok = _call_main("pcb_level3_engine_32", main_kwargs={})
    return 0 if ok else 2


def cmd_alerts(args):
    _ensure_out()
    data_in = _pick_data_path(preferred=args.data, allow_clean=True)
    ok = _call_main(
        "pcb_alerts_level28",
        main_kwargs={"data_csv_path": data_in, "insights_path": OUT_INSIGHTS_L2},
    )
    return 0 if ok else 2


def build_parser():
    p = argparse.ArgumentParser(
        prog="pcb",
        description="PCB — Unified CLI (2.6 -> 2.5 -> 2.8 -> 2.9 -> 3.2)",
    )
    p.add_argument("--config", default="pcb.json", help="Optional config JSON (default: pcb.json).")

    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("init", help="Initialize out/ folder and optional scaffolding files")
    sp.add_argument("--scaffold-trials", action="store_true", help="Create empty out/experiment_results.csv")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("run", help="Run full pipeline: 2.6 -> 2.5 -> 2.8 -> 2.9(plan/eval) -> 3.2")
    sp.add_argument("--data", default=None, help="Data path (default: out/data_clean.csv if exists else data.csv).")
    sp.add_argument("--skip-26", action="store_true", dest="skip_26", help="Skip Level 2.6")
    sp.add_argument("--skip-25", action="store_true", dest="skip_25", help="Skip Level 2.5")
    sp.add_argument("--skip-28", action="store_true", dest="skip_28", help="Skip Level 2.8")
    sp.add_argument("--skip-29-plan", action="store_true", dest="skip_29_plan", help="Skip Level 2.9 plan")
    sp.add_argument("--skip-29-eval", action="store_true", dest="skip_29_eval", help="Skip Level 2.9 eval")
    sp.add_argument("--skip-32", action="store_true", dest="skip_32", help="Skip Level 3.2")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("plan", help="Generate out/experiment_plan.csv (Level 2.9 plan)")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("log", help="Append one trial row into out/experiment_results.csv (Level 2.9 log)")
    sp.add_argument("--insight-id", required=True, dest="insight_id", help="Insight ID (e.g., I2-00001)")
    sp.add_argument("--action-name", default="", dest="action_name", help="Optional; auto-fill from plan if empty")
    sp.add_argument("--date", default="", help="Optional date YYYY-MM-DD")
    sp.add_argument("--t-index", default=None, type=int, dest="t_index", help="Optional t_index (overrides date)")
    sp.add_argument("--adherence-flag", default=1, type=int, dest="adherence_flag", help="0/1 performed?")
    sp.add_argument("--dose", default="", help="Optional dose/intensity string")
    sp.add_argument("--notes", default="", help="Optional notes")
    sp.add_argument("--data", default=None, help="Optional data path (for t_index resolution)")
    sp.set_defaults(func=cmd_log)

    sp = sub.add_parser("eval", help="Compute out/experiment_summary_level29.csv (Level 2.9 eval)")
    sp.add_argument("--data", default=None, help="Optional data path")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("alerts", help="Compute out/alerts_today_level28.csv (Level 2.8)")
    sp.add_argument("--data", default=None, help="Optional data path")
    sp.set_defaults(func=cmd_alerts)

    sp = sub.add_parser("validate", help="Run Level 3.2 counterfactual validation only")
    sp.set_defaults(func=cmd_validate)

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[pcb] interrupted.")
        return 130
    except Exception:
        print("[pcb] ERROR: unexpected exception")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
