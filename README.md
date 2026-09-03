# SWPL — Skew-Phase Law

A data structure selection law under Zipf-skewed workloads.

**Now on PyPI: `pip install swpl`**  |  [pypi.org/project/swpl](https://pypi.org/project/swpl/)  |  [GitHub](https://github.com/ZhangYangyi03/swpl)

---

## The Law

> Data structure performance under Zipf(α) access is not a continuous degradation — it exhibits **sharp phase boundaries** in the (α, w) plane, and any fixed structure is provably suboptimal in at least one phase.

### Critical threshold: α = 1

The threshold α = 1 is the Riemann ζ convergence point:

| α | ζ(α) | Hot set | Best structure |
|---|------|---------|----------------|
| α ≤ 1 | diverges | Θ(n) — all keys are "hot" | Static B-tree / hash |
| α > 1 | finite | Θ(δ^{-1/(α-1)}) | Adaptive (splay, ζ-Trie) |
| w ≥ 0.5 | — | — | LSM / write-optimised |

### Theorem (SWPL Crossover)

For any comparison-based dictionary on n keys with Zipf(α) access:

- **α > 1 + ε** → adaptive structure (splay) dominates
- **α < 1 − ε** → static structure (B-tree) dominates
- **ε → 0** as n → ∞

The proof follows from the **hot-set scaling lemma**: a hot set of size
M(δ) = Θ([δ(α−1)ζ(α)]^{-1/(α−1)}) covers (1−δ) of accesses. Splay achieves
O(log M), while a static tree must pay Ω(log n) regardless of pattern.

---

## ζ-Trie: the optimal structure

**ProbZetaTrie** is a probability-mass-partitioned trie: each node splits its
key subspace into b groups of **equal cumulative probability mass** (not equal
rank count). Under Zipf(α>1), hot keys get tiny buckets at shallow depth.

Measured on n=8192, b=8, α=1.5:

| depth | keys | probability mass |
|-------|------|-----------------|
| 0 | 2 keys (0.02%) | **52.3%** |
| 1 | 11 (0.1%) | 27.6% |
| 2+ | rest | 20.1% |

**Expected search depth = 0.88 comparisons** — matching the SWPL lower bound
O(log_b M_δ). Compare: static B-tree needs ~13, splay ~13.

```python
from swpl import ProbZetaTrie
zt = ProbZetaTrie(keys, b=8, alpha=1.5)
zt.expected_depth()   # 0.88
```

---

## Install

```bash
pip install swpl
```

## Quick start

```python
from swpl import recommend, estimate_alpha

# Direct recommendation
rec = recommend(alpha=1.4, write_ratio=0.1, ordered=True)
print(rec["winner"])     # "Adaptive tree (splay / working-set tree)"
print(rec["reason"])

# Estimate α from access log
from pathlib import Path
lines = Path("access.log").read_text().splitlines()
result = analyze_workload(lines, ordered=True)
print(f"Estimated α: {result['estimated_alpha']}")
print(f"Recommended: {result['recommendation']['winner']}")
```

## CLI

```bash
# Analyze a workload
swpl analyze access.log

# Direct recommendation from parameters
swpl recommend --alpha 1.5 --ordered

# Generate phase diagrams
swpl plot
```

## Citation

If you use SWPL in your research, cite as:

```bibtex
@misc{zhang2026swpl,
  title = {SWPL: A Skew–Phase Law for Data Structure Selection},
  author = {Zhang, Yangyi},
  year  = {2026},
  howpublished = {\url{https://github.com/ZhangYangyi03/swpl}}
}
```

---

## Repository structure

```
swpl/
├── pyproject.toml          # Build config
├── LICENSE                 # MIT
├── README.md
├── data/                   # Experimental data
└── swpl/
    ├── __init__.py
    ├── cli.py              # CLI interface
    ├── recommend.py        # Core: Theorem + MLE α estimation + recommendation
    ├── zeta_trie.py        # ProbZetaTrie (optimal) + RankZetaTrie + benchmark
    └── plot.py             # Phase diagram / hot-set / crossover plots
```