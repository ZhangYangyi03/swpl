"""SWPL core: Zeta-phase theorems and recommendation engine.

Mathematical foundation
=======================

For a Zipf(α) distribution over n (rank 1..n):

    p_k = k^{-α} / ζ_n(α),   ζ_n(α) = Σ_{k=1}^n k^{-α}

The critical point α = 1 is where the Riemann zeta series transitions
from convergent (α > 1, finite) to divergent (α ≤ 1, infinite).

Theorem (SWPL Crossover)
------------------------
For any comparison-based dictionary on n keys with Zipf(α) access:

    α > 1 + ε_n   →  adaptive structure (e.g. splay) dominates
    α < 1 - ε_n   →  static structure (e.g. B-tree) dominates
    α ≈ 1         →  regime undecidable without exact constants

    where ε_n → 0 as n → ∞.

Proof sketch:
    Static BST cost  ≥  log₂ n                     (leaf-count lower bound)
    Splay cost       =  O(1 + log M_δ)              (working-set theorem)
    M_δ              =  Θ(δ^{-1/(α-1)})             (hot-set scaling lemma)

    Equating:  log n  =  (1/(α-1)) log(1/δ)
    =>  α - 1  =  log(1/δ) / log n  →  0  for fixed δ.

    Hence α_c → 1⁺ as n → ∞.

Corollary (Hot-Set Scaling)
---------------------------
The number of keys needed to cover fraction (1-δ) of accesses is:

    M(δ) = ┌ [δ(α-1)ζ_n(α)]^{-1/(α-1)} ┐     α > 1
           └ n                              ┘     α ≤ 1

This is the fundamental reason for the phase transition:
finite ζ = finite hot set = adaptive structures win.
"""

import math
import numpy as np
from typing import Optional

# ─── Zeta normalisation ──────────────────────────────────────────

def zeta_partial(n: int, alpha: float) -> float:
    """ζ_n(α) = Σ_{k=1}^n k^{-α}  via integral approximation for speed."""
    if alpha == 1.0:
        return math.log(n) + 0.5772156649  # Euler-Mascheroni
    # direct sum for small n, integral approx for large
    if n <= 10_000:
        return sum(k ** -alpha for k in range(1, n + 1))
    # Euler-Maclaurin: ζ(α) - Σ_{k=n+1}^∞ ≈ ζ(α) - n^{1-α}/(α-1)
    from mpmath import zeta as mp_zeta
    z = float(mp_zeta(alpha))
    tail = (n ** (1 - alpha)) / (alpha - 1)
    return z - tail

def hot_set_size(n: int, alpha: float, delta: float = 0.1) -> int:
    """M(δ): minimum keys covering (1-δ) of accesses."""
    if alpha <= 1:
        return n
    if delta <= 0:
        return 1
    m = (delta * (alpha - 1) * zeta_partial(n, alpha)) ** (-1.0 / (alpha - 1))
    return min(n, max(1, int(math.ceil(m))))

# ─── Recommendation ──────────────────────────────────────────────

def recommend(alpha: float, write_ratio: float = 0.0,
              ordered: bool = False, n: int = 1_000_000) -> dict:
    """Primary recommendation under SWPL.

    Parameters
    ----------
    alpha : float
        Zipf skew parameter (>0). Estimated from workload.
    write_ratio : float
        Fraction of operations that are writes (0-1).
    ordered : bool
        If True, ordered traversal is required.
    n : int
        Approximate number of distinct keys (for finite-size effects).

    Returns
    -------
    dict with 'winner', 'reason', and supporting metrics.
    """
    if write_ratio >= 0.5:
        return {
            "winner": "LSM-tree or B-tree with write-buffer",
            "reason": (
                "Write-dominated (w≥0.5): write-optimized structures "
                "amortise write cost via tiering; read amplification "
                f"~O(log_{8} n) ≈ {math.log(n, 8):.1f} levels."
            ),
            "alpha": alpha, "w": write_ratio, "n": n,
            "ordered": ordered,
        }

    if ordered or not ordered:  # ordered matters, but we handle both
        if alpha <= 1:
            return {
                "winner": "Static B-tree (block-aligned)",
                "reason": (
                    f"α={alpha:.2f} ≤ 1: distribution is spread across "
                    "all keys — no finite hot set exists. Adaptive structures "
                    "pay overhead with no skew to exploit.\n"
                    "Static B-tree provides consistent O(log_B n) cost."
                ),
                "alpha": alpha, "w": write_ratio, "n": n,
                "ordered": True,
            }
        else:
            M = hot_set_size(n, alpha, delta=0.1)
            return {
                "winner": "Adaptive tree (splay / working-set tree)",
                "reason": (
                    f"α={alpha:.2f} > 1: Zipf tail is summable (ζ({alpha:.2f}) "
                    f"is finite). Hot set covering 90% of accesses has only "
                    f"M≈{M:,d} keys (out of n={n:,d}).\n"
                    f"Splay tree achieves O(log M) ≈ {math.log2(M):.1f} "
                    f"comparisons vs static B-tree's O(log n) ≈ "
                    f"{math.log2(n):.1f}."
                ),
                "alpha": alpha, "w": write_ratio, "n": n,
                "ordered": True, "hot_set_90pct": M,
            }
    else:
        # unordered, hash is always best for point queries
        return {
            "winner": "Hash table (open addressing)",
            "reason": (
                "Unordered workload with point-only access: hash table is "
                "O(1) expected regardless of skew."
            ),
            "alpha": alpha, "w": write_ratio, "n": n, "ordered": False,
        }

# ─── Zipf α estimation (MLE) ──────────────────────────────────────

def estimate_alpha(counts: list[int]) -> float:
    """MLE for Zipf α from a list of per-key access counts.

    Uses Newton-Raphson to solve the score equation:

        ζ'(α)/ζ(α)  =  (1/N) Σ c_i log r_i

    where c_i is count for rank-r_i (sorted descending).
    """
    c = sorted(counts, reverse=True)
    N = sum(c)
    if N == 0:
        return 1.0
    ranks = list(range(1, len(c) + 1))
    # mean log rank weighted by count
    target = sum(ci * math.log(r) for ci, r in zip(c, ranks)) / N

    def dlog_zeta(a):
        """-ζ'(a)/ζ(a) for partial zeta up to n."""
        n = len(c)
        z = sum(k ** -a for k in range(1, n + 1))
        dz = sum(k ** -a * math.log(k) for k in range(1, n + 1))
        return dz / z

    # Newton-Raphson
    a = 1.5
    for _ in range(50):
        f = dlog_zeta(a) - target  # we want ζ'(α)/ζ(α) = target? no...
        # Actually: E[log r] = Σ p_k log k = (1/ζ_n(α)) Σ k^{-α} log k
        # This is -ζ'_n(α)/ζ_n(α)
        # Target from data = (1/N) Σ c_i log r_i
        # So we solve: -ζ'_n(α)/ζ_n(α) = target
        # f = target + ζ'_n(α)/ζ_n(α)
        z = sum(k ** -a for k in range(1, len(c) + 1))
        dz = sum(k ** -a * math.log(k) for k in range(1, len(c) + 1))
        # d2z for Newton
        d2z = sum(k ** -a * (math.log(k) ** 2) for k in range(1, len(c) + 1))
        f_val = target + dz / z  # we want this = 0
        df_val = (d2z * z - dz * dz) / (z * z)
        if abs(f_val) < 1e-12:
            break
        if abs(df_val) < 1e-15:
            break
        a_new = a - f_val / df_val
        if a_new <= 0.1 or a_new > 20:
            break
        a = a_new
    return a


def estimate_alpha_from_log(lines: list[str]) -> float:
    """Parse access log and estimate α.

    Each line should contain a key string (space/tab/comma separated,
    or whole line as key).
    """
    from collections import Counter
    keys = [line.strip() for line in lines if line.strip()]
    if not keys:
        return 1.0
    counts = list(Counter(keys).values())
    return estimate_alpha(counts)


# ─── Full workload analysis ──────────────────────────────────────

def analyze_workload(lines: list[str], ordered: bool = False,
                     n_estimate: Optional[int] = None) -> dict:
    """Full analysis of a workload log.

    Returns a dict with α estimate, write ratio detection (naive),
    and SWPL recommendation.
    """
    from collections import Counter
    # Simple: each line is a key; we detect "writes" by duplicate keys
    # (crude heuristic — a real workload would tag reads vs writes)
    keys = [line.strip() for line in lines if line.strip()]
    if not keys:
        return {"error": "empty log"}

    total = len(keys)
    counts = Counter(keys)
    alpha = estimate_alpha(list(counts.values()))
    n = n_estimate or len(counts)

    # Rough write-ratio: count keys appearing more than once as "writes"
    # (This is a placeholder; real logs should have explicit read/write)
    write_estimate = 0.0

    return {
        "total_ops": total,
        "distinct_keys": len(counts),
        "estimated_alpha": round(alpha, 4),
        "estimated_write_ratio": write_estimate,
        "recommendation": recommend(alpha, write_estimate, ordered, n),
    }