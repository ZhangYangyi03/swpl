"""swpl — Skew–Phase Law for data structure selection.

Given a workload's Zipf skew (α) and write ratio (w),
recommend the optimal ordered or unordered data structure.
"""

from .recommend import recommend, estimate_alpha, estimate_alpha_from_log, analyze_workload
from .zeta_trie import ProbZetaTrie, RankZetaTrie, benchmark as zeta_benchmark
from .plot import plot_phase_diagram

__version__ = "0.1.0"
__all__ = ["recommend", "estimate_alpha", "analyze_workload", "plot_phase_diagram",
           "ProbZetaTrie", "RankZetaTrie", "zeta_benchmark"]