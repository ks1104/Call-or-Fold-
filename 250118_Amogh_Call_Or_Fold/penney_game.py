import random
from collections import Counter

random.seed(42)  # reproducibility

def play_one_race(seq_a, seq_b):
    window = ""
    for _ in range(3):
        window += random.choice("HT")

    while True:
        if window == seq_a:
            return "A"
        if window == seq_b:
            return "B"
        window = window[1:] + random.choice("HT")


def run_match(seq_a, seq_b, trials=100_000):
    results = Counter()
    for _ in range(trials):
        results[play_one_race(seq_a, seq_b)] += 1
    p_a = results["A"] / trials
    p_b = results["B"] / trials
    return p_a, p_b


def correlation(x, y):
    L = len(x)
    total = 0
    for k in range(L):
        if x[k:] == y[: L - k]:
            total += 2 ** (L - 1 - k)
    return total


def conway_odds(seq_a, seq_b):

    AA = correlation(seq_a, seq_a)
    AB = correlation(seq_a, seq_b)
    BB = correlation(seq_b, seq_b)
    BA = correlation(seq_b, seq_a)

    odds_a = BB - BA  
    odds_b = AA - AB  
    p_a = odds_a / (odds_a + odds_b)
    return p_a, odds_a, odds_b


#running the matches
MATCHUPS = [
    ("HTT", "HTH"),
    ("HTH", "THH"),
    ("THH", "TTH"),
]

TRIALS = 100_000


def main():
    print(f"Penney's Game simulation  ({TRIALS:,} races per matchup)\n")
    print(f"{'Matchup':<14}{'Emp. P(A)':>12}{'Exact P(A)':>13}{'Exact odds':>14}")
    print("-" * 53)

    rows = []
    for a, b in MATCHUPS:
        emp_a, emp_b = run_match(a, b, TRIALS)
        exact_a, oa, ob = conway_odds(a, b)
        rows.append((a, b, emp_a, emp_b, exact_a, oa, ob))
        print(f"{a} vs {b:<6}{emp_a:>12.4f}{exact_a:>13.4f}{f'{oa}:{ob}':>14}")

    print()
    for a, b, emp_a, emp_b, exact_a, oa, ob in rows:
        winner = a if exact_a > 0.5 else b
        print(f"  {a} vs {b}: {winner} wins "
              f"(empirical {max(emp_a, emp_b):.1%}, exact {max(exact_a, 1-exact_a):.1%})")

    with open("penney_results.txt", "w") as f:
        for a, b, emp_a, emp_b, exact_a, oa, ob in rows:
            f.write(f"{a},{b},{emp_a:.5f},{emp_b:.5f},{exact_a:.5f},{oa},{ob}\n")

    build_full_matrix_and_plot()


#visualisation of results
def build_full_matrix_and_plot():
    import numpy as np
    import matplotlib.pyplot as plt

    seqs = ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]
    n = len(seqs)

    M = np.full((n, n), 0.5)
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            if i != j:
                M[i, j] = conway_odds(a, b)[0]

    #heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(M, cmap="RdBu", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(seqs, rotation=45, ha="right")
    ax.set_yticklabels(seqs)
    ax.set_xlabel("Opponent (column)")
    ax.set_ylabel("Player (row)")
    ax.set_title("Penney's Game: exact P(row sequence beats column sequence)")
    for i in range(n):
        for j in range(n):
            val = M[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if abs(val - 0.5) > 0.22 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="P(row wins)")
    fig.tight_layout()
    fig.savefig("penney_heatmap.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [f"{a}\nvs\n{b}" for a, b in MATCHUPS]
    emp = []
    exa = []
    for a, b in MATCHUPS:
        ea, _ = run_match(a, b, TRIALS)
        emp.append(ea)
        exa.append(conway_odds(a, b)[0])
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, emp, w, label="Empirical P(A wins)", color="#4C72B0")
    ax.bar(x + w/2, exa, w, label="Exact P(A wins)", color="#DD8452")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("P(first sequence wins)")
    ax.set_ylim(0, 1)
    ax.set_title("Requested matchups: simulation vs exact theory")
    ax.legend()
    fig.tight_layout()
    fig.savefig("penney_matchups.png", dpi=130)
    plt.close(fig)

    print("\nNon-transitive cycle (each beats the next, exact P):")
    cycle = ["THH", "HHT", "HTT", "TTH", "THH"]
    for a, b in zip(cycle, cycle[1:]):
        p = conway_odds(a, b)[0]
        print(f"  {a} beats {b}: P = {p:.3f}")
    print("  ...back to the start: the relation loops, so there is no global winner.")


if __name__ == "__main__":
    main()
