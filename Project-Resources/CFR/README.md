# Counterfactual Regret Minimization: CFR, Deep CFR, and Neural Fictitious Self-Play

A research and implementation repository covering the core family of algorithms for solving imperfect-information extensive-form games, from tabular CFR to deep learning-based approximations.

---

## Overview

This repository implements and studies three related algorithms for computing Nash equilibria (or approximations thereof) in imperfect-information games:

| Algorithm | Representation | Scale | Key Idea |
|---|---|---|---|
| CFR | Tabular | Small games | Iterative regret minimization over information sets |
| Deep CFR | Neural network | Large games | Approximate CFR with learned value networks |
| Neller-Lanctot CFR | Tabular + variants | Educational / research | Pedagogically-oriented CFR variants from Neller & Lanctot (2013) |

The canonical application is poker (Kuhn, Leduc, Texas Hold'em), but the algorithms apply to any finite, two-player, zero-sum game in extensive form.

---

## Background

### Extensive-Form Games

An extensive-form game is a tree-structured game where players take actions sequentially, with possible chance nodes (e.g., card deals) and imperfect information (each player only sees their own cards). A player's *information set* groups all game states that are indistinguishable to that player given their observations.

A *strategy* maps each information set to a probability distribution over legal actions. A *Nash equilibrium* is a strategy profile where no player can unilaterally improve their expected payoff. In two-player zero-sum games, Nash equilibria correspond to minimax-optimal strategies.

### Regret

The central quantity in CFR is *regret*: the difference between the payoff a player could have achieved by playing a fixed action at every visit to an information set, versus the payoff actually obtained. Formally, for information set $I$ and action $a$:

$$R^T(I, a) = \sum_{t=1}^{T} \pi^{\sigma^t}_{-i}(I) \left( u_i(\sigma^t|_{I \to a}) - u_i(\sigma^t) \right)$$

CFR minimizes total regret across all information sets by updating strategies via *regret matching* at each iteration.

---

## Algorithm 1: CFR (Counterfactual Regret Minimization)

**Paper:** Zinkevich et al., *Regret Minimization in Games with Incomplete Information*, NeurIPS 2007.

### How It Works

CFR runs iteratively. At each iteration $t$:

1. **Traversal.** Walk the full game tree, computing counterfactual values for each information set under the current strategy profile $\sigma^t$.
2. **Regret update.** For each information set $I$ and action $a$, accumulate the instantaneous counterfactual regret.
3. **Regret matching.** Set the next-iteration strategy $\sigma^{t+1}(I, \cdot)$ proportional to the positive cumulative regrets:

$$\sigma^{t+1}(I, a) = \frac{\max(R^t(I,a), 0)}{\sum_{a'} \max(R^t(I,a'), 0)}$$

   If all cumulative regrets are non-positive, play uniformly.
4. **Average strategy.** The Nash equilibrium approximation is the time-average of $\sigma^1, \ldots, \sigma^T$, not the final iterate.

**Convergence.** After $T$ iterations, the average strategy is an $\varepsilon$-Nash equilibrium with $\varepsilon = O(1/\sqrt{T})$.

### Variants

- **CFR+** (Tammelin 2014): uses regret matching+ (floors negative regrets to zero each round) and alternating updates. Substantially faster convergence in practice.
- **Chance Sampling CFR (CS-CFR):** samples chance outcomes at each traversal, reducing per-iteration cost.
- **External Sampling / Outcome Sampling MCCFR:** Monte Carlo variants that sample the opponent's and/or chance's actions, enabling linear (not exponential) per-iteration cost.

### When to Use

CFR is practical when the game tree is small enough to traverse fully (information sets in the millions, not billions). Kuhn poker and Leduc Hold'em are standard benchmarks.

---

## Algorithm 2: Deep CFR

**Paper:** Brown et al., *Deep Counterfactual Regret Minimization*, ICML 2019.

### Motivation

Full-tree CFR is infeasible for large games (e.g., Texas Hold'em has ~$10^{14}$ decision points). Deep CFR replaces the tabular regret and strategy tables with neural networks that generalize across similar information sets.

### How It Works

Deep CFR maintains two types of neural networks:

- **Advantage networks** $V_\theta^i$: approximate the counterfactual advantage (regret) for player $i$ at each information set.
- **Strategy network** $\Pi_\phi$: approximates the average strategy.

Each CFR iteration proceeds as:

1. **Traversal with sampling.** Traverse the game tree via external sampling (sample opponent actions; walk all player $i$ actions).
2. **Collect advantage targets.** At each information set visited by player $i$, record $(s_I, \mathbf{a}_I, \Delta_I)$ where $\Delta_I(a) = v(a) - \bar{v}$ is the instantaneous advantage for each action.
3. **Train advantage network.** Fit $V_\theta^i$ on the collected buffer using supervised regression.
4. **Strategy target collection.** Record $(s_I, \sigma^t_I)$ for the strategy network.
5. **Train strategy network.** Fit $\Pi_\phi$ on the strategy buffer.

The key insight is that regret-matching is applied to network *outputs* rather than stored tables: the current strategy at any information set is recovered by applying regret matching to $V_\theta^i(s_I, \cdot)$ on the fly.

### Architecture Notes

- The information set is encoded as a fixed-length feature vector (card abstractions, betting history, position, stack sizes).
- Separate networks per player; networks are retrained periodically from a growing reservoir buffer.
- Reservoir sampling is used to maintain a size-bounded buffer that is representative of all past iterations, not just recent ones.

### When to Use

Deep CFR scales to games where the information set space is too large for tabular methods. It trades off exact convergence guarantees for practical scalability, and requires careful hyperparameter tuning (network capacity, buffer size, traversal count per iteration).

---

## Algorithm 3: Neller-Lanctot CFR

**Reference:** Neller & Lanctot, *An Introduction to Counterfactual Regret Minimization*, 2013.

This refers to the pedagogical framework and notation standardized by Todd Neller and Marc Lanctot in their tutorial paper, which has become the standard reference for teaching CFR and its variants. The repository includes implementations matching this paper's notation and examples closely.

### Coverage

The Neller-Lanctot treatment covers:

- **Vanilla CFR** with full game tree traversal, as described above.
- **Chance Sampling CFR:** at each chance node, sample a single outcome rather than enumerating all. Reduces per-iteration cost; preserves convergence in expectation.
- **External Sampling CFR:** opponent actions are also sampled; only the traversing player's actions are walked fully. Unbiased estimates; $O(|A|)$ per-iteration complexity in action count.
- **Outcome Sampling CFR:** a single terminal history is sampled per iteration using importance weights. Highest variance but cheapest per-iteration cost.

### Kuhn Poker Benchmark

All variants are benchmarked on Kuhn poker (3-card, 2-player, 1-raise), for which the analytic Nash equilibrium is known. This allows exact exploitability measurement and comparison of convergence rates across variants.

### Notation Reference

| Symbol | Meaning |
|---|---|
| $h$ | History (sequence of actions from root) |
| $I(h)$ | Information set containing history $h$ |
| $\sigma_i(I, a)$ | Strategy: probability player $i$ plays $a$ at $I$ |
| $\pi^\sigma(h)$ | Reach probability of $h$ under $\sigma$ |
| $\pi^\sigma_i(h)$ | Player $i$'s contribution to reach probability |
| $\pi^\sigma_{-i}(h)$ | Counterfactual reach (all players except $i$) |
| $u_i(\sigma)$ | Expected utility of player $i$ under $\sigma$ |
| $v_i(\sigma, I)$ | Counterfactual value of information set $I$ |
| $R^T_i(I, a)$ | Cumulative counterfactual regret at $(I, a)$ after $T$ iterations |

---

## Exploitability

All implementations are evaluated by *exploitability*: the gain a best-response opponent could achieve against the average strategy, expressed in milli-big-blinds per hand (mbb/h) or as a fraction of the game value. A Nash equilibrium has exploitability 0. Lower is better.

Exploitability is computed via a full best-response pass through the game tree (for tabular methods) or via sampled best response (for Deep CFR at scale).

---

## References

- Zinkevich, M., Johanson, M., Bowling, M., Piccione, C. (2007). *Regret Minimization in Games with Incomplete Information.* NeurIPS.
- Lanctot, M., Waugh, K., Zinkevich, M., Bowling, M. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games.* NeurIPS.
- Tammelin, O. (2014). *Solving Large Imperfect Information Games Using CFR+.* arXiv:1407.5042.
- Neller, T., Lanctot, M. (2013). *An Introduction to Counterfactual Regret Minimization.* Proc. Model AI Assignments.
- Brown, N., Lerer, A., Gross, S., Sandholm, T. (2019). *Deep Counterfactual Regret Minimization.* ICML.
- Brown, N., Sandholm, T. (2019). *Superhuman AI for multiplayer poker.* Science.
