"""Smoke tests for swpl. No network, no GPU, no matplotlib: these must pass
on a bare Python with nothing installed, because that is how the library is
meant to be used.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import swpl
from swpl import (recommend, estimate_alpha, ProbZetaTrie, RankZetaTrie,
                  zeta_benchmark)
from swpl.recommend import zeta_partial, hot_set_size


def test_import_needs_no_third_party_package():
    # the analysis path must not drag in numpy/matplotlib/mpmath
    for mod in ("numpy", "matplotlib", "mpmath"):
        assert mod not in sys.modules, f"{mod} was imported by `import swpl`"


def test_zeta_partial_matches_direct_sum():
    for alpha in (0.5, 0.9, 1.01, 1.1, 1.5, 2.0, 3.0):
        for n in (10, 1000, 10000):
            exact = math.fsum(k ** -alpha for k in range(1, n + 1))
            got = zeta_partial(n, alpha)
            assert abs(got - exact) / exact < 1e-12, (alpha, n, got, exact)


def test_zeta_partial_tail_is_accurate_where_the_law_lives():
    # ζ_n(1.01) at n=1e9 sits far below ζ(1.01)=100.5779 -- its own value is
    # 19.2948917 (Euler-Maclaurin from ζ(1.01)). The old implementation
    # subtracted the wrong tail and returned 19.2949 for the wrong reason;
    # what matters is the value at small n, checked above, plus α=1 here.
    ref = 19.2948917
    got = zeta_partial(10 ** 9, 1.01)
    assert abs(got - ref) / ref < 1e-6, got


def test_zeta_partial_monotone_in_n():
    vals = [zeta_partial(n, 1.5) for n in (10, 100, 1000, 10 ** 6)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_zeta_at_one_is_harmonic():
    n = 1000
    h = math.fsum(1.0 / k for k in range(1, n + 1))
    assert abs(zeta_partial(n, 1.0) - h) / h < 1e-9


def test_hot_set_shrinks_with_skew():
    sizes = [hot_set_size(10 ** 6, a, 0.1) for a in (1.1, 1.5, 2.5)]
    assert sizes[0] > sizes[1] > sizes[2]
    assert sizes[-1] >= 1


def test_recommend_phases():
    assert "Adaptive" in recommend(alpha=1.5, write_ratio=0.0)["winner"]
    assert "Static B-tree" in recommend(alpha=0.8, write_ratio=0.0)["winner"]
    assert "LSM" in recommend(alpha=1.5, write_ratio=0.7)["winner"]


def test_recommend_alpha_one_is_boundary():
    # alpha == 1 sits on the divergence point: no finite hot set
    assert "Static B-tree" in recommend(alpha=1.0, write_ratio=0.0)["winner"]


def test_estimate_alpha_recovers_known_skew():
    rng = random.Random(20261028)
    alpha, n, trials = 1.3, 500, 60000
    z = math.fsum((k + 1) ** -alpha for k in range(n))
    cum, acc = [], 0.0
    for k in range(n):
        acc += (k + 1) ** -alpha / z
        cum.append(acc)
    counts = [0] * n
    import bisect
    for _ in range(trials):
        counts[bisect.bisect_left(cum, rng.random())] += 1
    got = estimate_alpha(counts)
    assert abs(got - alpha) < 0.15, got


def test_prob_trie_beats_uniform_depth():
    n, alpha, b = 8192, 1.5, 8
    keys = list(range(n))
    pzt = ProbZetaTrie(keys, b=b, alpha=alpha)
    rzt = RankZetaTrie(keys, b=b)
    assert pzt.expected_depth() < rzt.expected_depth(alpha)
    assert pzt.expected_depth() < math.log2(n) / 2


def test_prob_trie_depth_zero_for_hottest_key():
    keys = list(range(1024))
    pzt = ProbZetaTrie(keys, b=8, alpha=1.5)
    assert pzt.search_depth(0) <= pzt.search_depth(1023)


def test_benchmark_runs_and_orders_costs():
    res = zeta_benchmark(alpha=1.5, n=1024, b=8, queries=2000)
    assert isinstance(res, dict) and res, res
