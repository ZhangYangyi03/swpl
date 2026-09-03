#!/usr/bin/env python3
"""SWPL CLI: analyze workloads and recommend data structures.

Usage:
    swpl analyze <logfile>          Estimate Zipf α + recommend
    swpl recommend [--alpha A] [--w W] [--ordered] [--n N]
    swpl plot [--type TYPE]         Generate phase diagram / hot-set / crossover
    swpl version
"""

import sys
import argparse
from pathlib import Path

from swpl import recommend, estimate_alpha, estimate_alpha_from_log, analyze_workload


def cmd_analyze(args):
    log_path = args.logfile
    if not Path(log_path).exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    result = analyze_workload(lines, ordered=args.ordered)
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print("=" * 60)
    print("SWPL Workload Analysis")
    print("=" * 60)
    print(f"  Total operations:     {result['total_ops']:,}")
    print(f"  Distinct keys:        {result['distinct_keys']:,}")
    print(f"  Estimated α:          {result['estimated_alpha']:.4f}")
    rec = result["recommendation"]
    print()
    print(f"  ▸ Recommended structure:  {rec['winner']}")
    print()
    print(f"  Reason:")
    for line in rec['reason'].strip().split('\n'):
        print(f"    {line}")
    print()
    if "hot_set_90pct" in rec:
        print(f"  Hot set (90% coverage):  {rec['hot_set_90pct']:,} keys")
    print("=" * 60)


def cmd_recommend(args):
    a = args.alpha
    w = args.w
    ordered = args.ordered
    n = args.n
    r = recommend(a, w, ordered, n)
    print(f"SWPL Recommendation for α={a:.2f}, w={w:.2f}, n={n:,d}")
    print(f"  ▸ {r['winner']}")
    print()
    for line in r["reason"].strip().split("\n"):
        print(f"  {line}")
    if "hot_set_90pct" in r:
        print(f"\n  Hot set (90% coverage): {r['hot_set_90pct']:,} keys")


def cmd_plot(args):
    from swpl.plot import plot_phase_diagram, plot_hot_set_scaling, plot_crossover
    if args.type == "phase":
        plot_phase_diagram(save=args.output or "swpl_phase_diagram.png")
    elif args.type == "hot":
        plot_hot_set_scaling(save=args.output or "swpl_hot_set.png")
    elif args.type == "crossover":
        plot_crossover(save=args.output or "swpl_crossover.png")
    elif args.type == "all":
        plot_phase_diagram(save="swpl_phase_diagram.png")
        plot_hot_set_scaling(save="swpl_hot_set.png")
        plot_crossover(save="swpl_crossover.png")
    else:
        print(f"Unknown plot type: {args.type}")


def main():
    parser = argparse.ArgumentParser(
        prog="swpl",
        description="Skew–Phase Law (SWPL): data structure selection under Zipf workloads"
    )
    sub = parser.add_subparsers(title="commands", dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a workload log file")
    p_analyze.add_argument("logfile", help="Path to access log (one key per line)")
    p_analyze.add_argument("--ordered", action="store_true",
                           help="Require ordered traversal")

    # recommend
    p_rec = sub.add_parser("recommend", help="Direct recommendation from parameters")
    p_rec.add_argument("--alpha", type=float, default=1.2, help="Zipf α (default: 1.2)")
    p_rec.add_argument("--w", type=float, default=0.0, help="Write ratio (default: 0.0)")
    p_rec.add_argument("--ordered", action="store_true", help="Require ordered traversal")
    p_rec.add_argument("--n", type=int, default=1_000_000, help="Key count (default: 1M)")

    # plot
    p_plot = sub.add_parser("plot", help="Generate SWPL phase diagram")
    p_plot.add_argument("--type", choices=["phase", "hot", "crossover", "all"],
                        default="all", help="Plot type (default: all)")
    p_plot.add_argument("--output", "-o", help="Output path (default: auto-named)")

    # version
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "recommend":
        cmd_recommend(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "version":
        from swpl import __version__
        print(f"swpl v{__version__}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()