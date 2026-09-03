# SWPL — Skew-Phase Law

A data structure selection law under Zipf-skewed workloads.

---

## The Law

> Data structure performance under Zipf(α) access is not a continuous degradation — it exhibits **sharp phase boundaries** in the (α, w) plane, and any fixed structure is provably suboptimal in at least one phase.

### Critical threshold: α = 1

The threshold α = 1 is the Riemann ζ convergence point:

| α | ζ(α) | Hot set | Best structure |
|---|------|---------|----------------|
| α ≤ 1 | diverges | Θ(n) — all keys are "hot" | Static B-tree / hash |
| α > 1 | finite | Θ(δ^{-1/(α-1)}) | Adaptive (splay, working-set tree) |
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
  howpublished = {\url{https://github.com/yz1571/swpl}}
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
    └── plot.py             # Phase diagram / hot-set / crossover plots
```