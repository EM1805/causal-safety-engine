# FILE: run_pcb.py
# Python 3.7 compatible
#
# PCB — Unified CLI (pipeline runner)
# 2.6 -> 2.5 -> 2.8 -> 2.9(plan/eval optional) -> 3.2 (optional; requires experiment_results.csv)
#
# Usage:
#   python src/run_pcb.py run --data data.csv
#   python src/run_pcb.py run --data out/data_clean.csv --skip-28
#   python src/run_pcb.py plan
#   python src/run_pcb.py log --insight_id I2-00001 --notes "did it"
#   python src/run_pcb.py eval
#   python src/run_pcb.py level32   (requires experiment_results.csv)
#
import os
import sys
import argparse
from pcb_graph_export import export_personal_causal_graph

# Add src to path for local imports
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pcb_data_level26
import pcb_insights_level25
import pcb_alerts_level28
import pcb_experiments_level29
import pcb_level3_engine_32

OUT_DIR = "out"

def _copy_if(src_path, dst_path):
    if src_path and os.path.exists(src_path):
        import shutil
        shutil.copyfile(src_path, dst_path)

def cmd_run(args):
    data_path = args.data
    # Level 2.6 (creates out/data_clean.csv)
    pcb_data_level26.main(data_csv_path=data_path)

    clean_path = os.path.join(OUT_DIR, "data_clean.csv")
    if os.path.exists(clean_path) and args.use_clean_for_next:
        # Use clean for subsequent levels
        data_for_next = clean_path
    else:
        data_for_next = data_path

    # Level 2.5 (insights)
    pcb_insights_level25.main(data_csv_path=data_for_next)

    # Level 2.8 (alerts)
    if not args.skip_28:
        pcb_alerts_level28.main(data_csv_path=data_for_next, insights_path=os.path.join(OUT_DIR, "insights_level2.csv"))

    # Level 2.9 plan/eval (optional)
    if args.run_plan:
        pcb_experiments_level29.main(["plan"])
    if args.run_eval:
        pcb_experiments_level29.main(["eval"])

    # Level 3.2 (optional; needs experiment_results.csv)
    if args.run_32:
        pcb_level3_engine_32.main(data_path=data_for_next)

    print("\n=== PIPELINE DONE ===")
    print("Out dir:", OUT_DIR)

def cmd_plan(args):
    pcb_experiments_level29.main(["plan"])

def cmd_log(args):
    argv = ["log", "--insight_id", args.insight_id, "--adherence_flag", str(args.adherence_flag)]
    if args.action_name:
        argv += ["--action_name", args.action_name]
    if args.date:
        argv += ["--date", args.date]
    if args.t_index is not None:
        argv += ["--t_index", str(args.t_index)]
    if args.dose:
        argv += ["--dose", args.dose]
    if args.notes:
        argv += ["--notes", args.notes]
    if args.data:
        argv += ["--data", args.data]
    pcb_experiments_level29.main(argv)

def cmd_eval(args):
    argv = ["eval"]
    if args.data:
        argv += ["--data", args.data]
    pcb_experiments_level29.main(argv)

def cmd_level32(args):
    pcb_level3_engine_32.main(data_path=args.data)

def build_argparser():
    p = argparse.ArgumentParser(prog="run_pcb.py", description="PCB unified CLI (local-first).")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run pipeline: 2.6 -> 2.5 -> 2.8 -> (optional 2.9 plan/eval) -> (optional 3.2)")
    p_run.add_argument("--data", default=None, help="Input data path (default: data.csv or out/demo_data.csv)")
    p_run.add_argument("--use-clean-for-next", action="store_true", help="Use out/data_clean.csv as input for next levels")
    p_run.add_argument("--skip-28", action="store_true", help="Skip Level 2.8 alerts")
    p_run.add_argument("--run-plan", action="store_true", help="Run Level 2.9 plan at end")
    p_run.add_argument("--run-eval", action="store_true", help="Run Level 2.9 eval at end")
    p_run.add_argument("--run-32", action="store_true", help="Run Level 3.2 at end (requires out/experiment_results.csv)")
    p_run.set_defaults(func=cmd_run)

    p_plan = sub.add_parser("plan", help="Generate experiment plan (2.9 plan)")
    p_plan.set_defaults(func=cmd_plan)

    p_log = sub.add_parser("log", help="Append one trial row (2.9 log)")
    p_log.add_argument("--insight_id", required=True)
    p_log.add_argument("--action_name", default="")
    p_log.add_argument("--date", default="")
    p_log.add_argument("--t_index", type=int, default=None)
    p_log.add_argument("--adherence_flag", type=int, default=1)
    p_log.add_argument("--dose", default="")
    p_log.add_argument("--notes", default="")
    p_log.add_argument("--data", default=None)
    p_log.set_defaults(func=cmd_log)

    p_eval = sub.add_parser("eval", help="Evaluate trials (2.9 eval)")
    p_eval.add_argument("--data", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_32 = sub.add_parser("level32", help="Run Level 3.2 counterfactual validation (requires out/experiment_results.csv)")
    p_32.add_argument("--data", default=None)
    p_32.set_defaults(func=cmd_level32)

    return p

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_argparser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
