"""Phase diagram plotting for SWPL."""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .recommend import hot_set_size

__all__ = ["plot_phase_diagram", "plot_hot_set_scaling", "plot_crossover"]


def plot_phase_diagram(n: int = 8192, save: str = "swpl_phase_diagram.png"):
    """(α, w) phase diagram for ordered structures."""
    alphas = np.linspace(0.3, 4.0, 60)
    writes = np.linspace(0, 1, 50)
    AA, WW = np.meshgrid(alphas, writes)

    # Label each cell: treap=0, splay=1, lsm=2
    Z = np.zeros_like(AA)
    # Cost functions (comparison-model)
    for i, a in enumerate(alphas):
        for j, w in enumerate(writes):
            H = math.log2(n)  # treap: O(log n)
            M = hot_set_size(n, a, 0.1)
            S = math.log2(max(M, 2))  # splay: O(log M)
            L = 1 + w * 1.0 + (1 - w) * math.log(n, 8)  # LSM approx
            costs = [H, S, L]
            Z[j, i] = np.argmin(costs)

    labels = ["Static B-tree", "Splay (adaptive)", "LSM-tree"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(AA, WW, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                          colors=colors, alpha=0.85)
    # Patch legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc="upper right", fontsize=11)

    # Critical lines
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label=r"$\alpha = 1$ (Riemann $\zeta$ divergence)")
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.5,
               label="$w = 0.5$ write threshold")

    ax.set_xlabel(r"Zipf skew $\alpha$", fontsize=13)
    ax.set_ylabel(r"Write ratio $w$", fontsize=13)
    ax.set_title("SWPL Phase Diagram for Ordered Structures", fontsize=14)
    ax.set_xlim(0.3, 4.0)
    ax.set_ylim(0, 1)

    text = (
        r"$\alpha > 1$: $\zeta(\alpha)$ finite $\Rightarrow$ hot set exists"
        "\n"
        r"$\alpha \leq 1$: $\zeta(\alpha)$ diverges $\Rightarrow$ uniform spread"
    )
    ax.text(0.35, 0.5, text, transform=ax.transData,
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save, dpi=200)
    plt.close()
    print(f"Phase diagram saved to {save}")
    return save


def plot_hot_set_scaling(n: int = 1_000_000, save: str = "swpl_hot_set.png"):
    """Hot-set size M(δ) vs α for various δ."""
    alphas = np.linspace(1.01, 3.0, 100)
    deltas = [0.01, 0.05, 0.1, 0.2]

    fig, ax = plt.subplots(figsize=(8, 5))
    for d in deltas:
        M = [hot_set_size(n, a, delta=d) for a in alphas]
        ax.plot(alphas, M, label=rf"$\delta={d:.2f}$ (covers ${(1-d)*100:.0f}\%$)",
                linewidth=2)

    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"Zipf skew $\alpha$", fontsize=13)
    ax.set_ylabel(r"Hot set size $M(\delta)$", fontsize=13)
    ax.set_title(r"Hot-Set Scaling: $M(\delta) \sim [\delta(\alpha-1)\zeta(\alpha)]^{-1/(\alpha-1)}$",
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=200)
    plt.close()
    print(f"Hot-set scaling saved to {save}")
    return save


def plot_crossover(save: str = "swpl_crossover.png"):
    """Empirical crossover from the n=8192 simulation."""
    import json, os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             "swpl_phase_ordered.json")
    if not os.path.exists(data_path):
        print(f"Crossover data not at {data_path}")
        return

    with open(data_path) as f:
        raw = json.load(f)

    alphas = sorted(set(p["alpha"] for p in raw))
    writes = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, a in enumerate(alphas):
        if idx >= len(axes):
            break
        row = [p for p in raw if p["alpha"] == a]
        tr = [p["treap"] for p in row]
        sp = [p["splay"] for p in row]
        ls = [p["lsm"] for p in row]
        ax = axes[idx]
        ax.plot(writes, tr, "o-", label="treap (static)", color="#4c72b0")
        ax.plot(writes, sp, "s-", label="splay (adaptive)", color="#dd8452")
        ax.plot(writes, ls, "^-", label="LSM", color="#55a868")
        ax.set_title(rf"$\alpha={a:.2f}$", fontsize=11)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
        ax.set_xlabel("w")
        if idx // 4 == 0:
            ax.set_ylabel("comparison cost")

    plt.suptitle("SWPL Crossover: ordered-structure cost vs write ratio", fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=200)
    plt.close()
    print(f"Crossover plot saved to {save}")
    return save