"""swpl — Skew–Phase Law for data structure selection.

Given a workload's Zipf skew (α) and write ratio (w),
recommend the optimal ordered or unordered data structure.

Plotting is imported lazily so that `import swpl` needs only numpy, not
matplotlib: the analysis path should work on a machine where matplotlib is
missing or broken.
"""

from .recommend import recommend, estimate_alpha, estimate_alpha_from_log, analyze_workload
from .zeta_trie import ProbZetaTrie, RankZetaTrie, benchmark as zeta_benchmark

__version__ = "0.1.1"

__all__ = ["recommend", "estimate_alpha", "estimate_alpha_from_log",
           "analyze_workload", "plot_phase_diagram",
           "ProbZetaTrie", "RankZetaTrie", "zeta_benchmark"]


def plot_phase_diagram(*args, **kwargs):
    """Lazy proxy: pulls in matplotlib only when a plot is actually asked for."""
    from .plot import plot_phase_diagram as _f
    return _f(*args, **kwargs)
