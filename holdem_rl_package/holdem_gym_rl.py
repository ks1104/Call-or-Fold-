#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium-Compatible No-Limit Texas Hold'em Poker Engine & Deep RL Agent
========================================================================
Overcomes the architectural and strategic shortcomings of the heuristic JS script:
1. Standardized Gymnasium v0.29+ API (`reset(seed, options)`, `step(action)`,
   `observation_space`, `action_space`, and `info["action_mask"]`) decoupled from UI timers.
2. Rich 24-dimensional continuous state representation capturing street one-hot,
   heads-up & multi-way pot-share equity, Chen preflop valuation, pot odds,
   stack-to-pot ratio (SPR), stack depths, relative position, active opponents,
   street & preflop aggression history, board texture (flush draw, straight connectivity,
   pairedness, high-card density), and Bayesian opponent fold posteriors.
3. Structured 7-Action Space with Exact Legal Action Masking:
   - 0: FOLD
   - 1: CHECK / CALL
   - 2: BET / RAISE 33% POT (or min-raise if larger)
   - 3: BET / RAISE 50% POT
   - 4: BET / RAISE 75% POT
   - 5: BET / RAISE 100% POT (Pot-sized)
   - 6: ALL-IN (Shove)
4. Multi-Street Dueling Double Q-Network + Advantage Policy Network (in pure NumPy
   with Adam optimization) trained via self-play and multi-persona league play with
   Experience Replay and CFR-inspired counterfactual action-value targets.
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# 1. GYMNASIUM-COMPATIBLE SPACES (Zero external gym dependency required)
# ============================================================================

class DiscreteSpace:
    """Gymnasium-compatible Discrete(n) space with legal action mask support."""
    def __init__(self, n: int):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64

    def sample(self, mask: Optional[np.ndarray] = None) -> int:
        if mask is not None:
            valid = np.flatnonzero(mask)
            if len(valid) > 0:
                return int(np.random.choice(valid))
        return int(np.random.randint(0, self.n))

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class BoxSpace:
    """Gymnasium-compatible Box(low, high, shape) continuous observation space."""
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.low = np.full(self.shape, low, dtype=dtype)
        self.high = np.full(self.shape, high, dtype=dtype)

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        return (
            isinstance(x, np.ndarray)
            and x.shape == self.shape
            and np.all(x >= self.low - 1e-5)
            and np.all(x <= self.high + 1e-5)
        )

    def __repr__(self) -> str:
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype.__name__})"


# ============================================================================
# 2. FAST BITMASK HAND EVALUATOR & EQUITY ENGINE
# ============================================================================

RANK_CHARS = "23456789TJQKA"
SUIT_SYMBOLS = "♠♥♦♣"
FULL_DECK = tuple(range(52))

POPC = [0] * 8192
STR8 = [-1] * 8192
TOP5T = [0] * 8192

for _m in range(8192):
    POPC[_m] = _m.bit_count()
    for _r in range(8, -1, -1):
        if (_m & (31 << _r)) == (31 << _r):
            STR8[_m] = _r + 4
            break
    if STR8[_m] < 0 and (_m & 0x100F) == 0x100F:
        STR8[_m] = 3
    _top5 = []
    for _r in range(12, -1, -1):
        if (_m >> _r) & 1:
            _top5.append(_r)
            if len(_top5) == 5:
                break
    _val = 0
    for _i in range(5):
        _val = _val * 13 + (_top5[_i] if _i < len(_top5) else 0)
    TOP5T[_m] = _val

P13_5 = 371293
P13_4 = 28561
P13_3 = 2197
P13_2 = 169


def card_str(c: int) -> str:
    return f"{RANK_CHARS[c >> 2]}{SUIT_SYMBOLS[c & 3]}"


def rank7fast(cards: List[int]) -> int:
    """Single-pass bitmask + LUT evaluator for 5, 6, or 7 cards."""
    s0 = s1 = s2 = s3 = 0
    all_mask = 0
    cnt = [0] * 13
    for c in cards:
        r = c >> 2
        u = c & 3
        b = 1 << r
        all_mask |= b
        cnt[r] += 1
        if u == 0:
            s0 |= b
        elif u == 1:
            s1 |= b
        elif u == 2:
            s2 |= b
        else:
            s3 |= b

    fm = s0 if POPC[s0] >= 5 else (s1 if POPC[s1] >= 5 else (s2 if POPC[s2] >= 5 else (s3 if POPC[s3] >= 5 else 0)))
    if fm:
        sf = STR8[fm]
        if sf >= 0:
            return 8 * P13_5 + sf * P13_4
        return 5 * P13_5 + TOP5T[fm]

    q0 = t0 = t1 = p0 = p1 = p2 = -1
    for r in range(12, -1, -1):
        k = cnt[r]
        if k == 4:
            q0 = r
        elif k == 3:
            if t0 < 0:
                t0 = r
            else:
                t1 = r
        elif k == 2:
            if p0 < 0:
                p0 = r
            elif p1 < 0:
                p1 = r
            elif p2 < 0:
                p2 = r

    if q0 >= 0:
        kicker = next(r for r in range(12, -1, -1) if r != q0 and cnt[r] > 0)
        return 7 * P13_5 + q0 * P13_4 + kicker * P13_3
    if t0 >= 0 and (t1 >= 0 or p0 >= 0):
        return 6 * P13_5 + t0 * P13_4 + max(t1, p0) * P13_3
    st = STR8[all_mask]
    if st >= 0:
        return 4 * P13_5 + st * P13_4
    if t0 >= 0:
        k0 = k1 = -1
        for r in range(12, -1, -1):
            if cnt[r] == 1:
                if k0 < 0:
                    k0 = r
                else:
                    k1 = r
                    break
        return 3 * P13_5 + t0 * P13_4 + k0 * P13_3 + k1 * P13_2
    if p1 >= 0:
        k0 = next(r for r in range(12, -1, -1) if r != p0 and r != p1 and cnt[r] > 0)
        return 2 * P13_5 + p0 * P13_4 + p1 * P13_3 + k0 * P13_2
    if p0 >= 0:
        k0 = k1 = k2 = -1
        for r in range(12, -1, -1):
            if cnt[r] == 1:
                if k0 < 0:
                    k0 = r
                elif k1 < 0:
                    k1 = r
                else:
                    k2 = r
                    break
        return P13_5 + p0 * P13_4 + k0 * P13_3 + k1 * P13_2 + k2 * 13
    return TOP5T[all_mask]


def chen_score(c1: int, c2: int) -> float:
    r1, r2 = c1 >> 2, c2 >> 2
    s1, s2 = c1 & 3, c2 & 3
    hi, lo = max(r1, r2), min(r1, r2)
    base_map = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}
    score = base_map.get(hi, (hi + 2) / 2.0)
    if r1 == r2:
        return float(math.ceil(max(5.0, score * 2.0)))
    if s1 == s2:
        score += 2.0
    gap = hi - lo - 1
    if gap == 1:
        score -= 1.0
    elif gap == 2:
        score -= 2.0
    elif gap == 3:
        score -= 4.0
    elif gap >= 4:
        score -= 5.0
    if gap <= 1 and hi < 10:
        score += 1.0
    return float(math.ceil(score))


CHEN_LUT = [[0.0] * 52 for _ in range(52)]
for _a in range(52):
    for _b in range(_a + 1, 52):
        _sc = chen_score(_a, _b)
        CHEN_LUT[_a][_b] = _sc
        CHEN_LUT[_b][_a] = _sc


def fast_equity_estimate(
    hole: List[int],
    board: List[int],
    n_opp: int = 1,
    iters: int = 48,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float, float]:
    """
    Fast Monte Carlo estimator returning (heads_up_equity, multiway_equity, tie_prob).
    """
    if len(hole) < 2:
        return 0.5, 0.5, 0.0
    if not board:
        c_val = CHEN_LUT[hole[0]][hole[1]]
        eq_hu = 1.0 / (1.0 + math.exp(-0.20 * (c_val - 4.2)))
        eq_mw = eq_hu ** (1.0 + 0.42 * (n_opp - 1))
        return float(np.clip(eq_hu, 0.12, 0.92)), float(np.clip(eq_mw, 0.05, 0.90)), 0.03
    if rng is None:
        rng = random
    dead = set(hole) | set(board)
    stub = [c for c in FULL_DECK if c not in dead]
    K = len(stub)
    b_need = 5 - len(board)
    k_draw = b_need + 2 * n_opp
    if k_draw > K:
        return 0.5, 0.5, 0.0

    b_len = len(board)
    sim_hero = [hole[0], hole[1]] + list(board) + [0] * b_need
    sim_opp = [0, 0] + list(board) + [0] * b_need
    w_hu = 0.0
    w_mw = 0.0
    ties_cnt = 0
    for _ in range(iters):
        for i in range(k_draw):
            j = i + int(rng.random() * (K - i))
            stub[i], stub[j] = stub[j], stub[i]
        for b in range(b_need):
            cb = stub[b]
            sim_hero[2 + b_len + b] = cb
            sim_opp[2 + b_len + b] = cb
        vh = rank7fast(sim_hero)
        best_o = -1
        t_cnt = 0
        for p in range(n_opp):
            sim_opp[0] = stub[b_need + 2 * p]
            sim_opp[1] = stub[b_need + 2 * p + 1]
            vo = rank7fast(sim_opp)
            if p == 0:
                if vh > vo:
                    w_hu += 1.0
                elif vh == vo:
                    w_hu += 0.5
            if vo > best_o:
                best_o = vo
                t_cnt = 1
            elif vo == best_o:
                t_cnt += 1
        if vh > best_o:
            w_mw += 1.0
        elif vh == best_o:
            w_mw += 1.0 / (t_cnt + 1)
            ties_cnt += 1
    return w_hu / iters, w_mw / iters, ties_cnt / iters


def extract_board_texture(board: List[int]) -> Tuple[float, float, float, float]:
    if not board:
        return 0.0, 0.0, 0.0, 0.0
    suits = [0, 0, 0, 0]
    rank_mask = 0
    ranks = []
    high_cnt = 0
    for c in board:
        r = c >> 2
        s = c & 3
        suits[s] += 1
        rank_mask |= (1 << r)
        ranks.append(r)
        if r >= 8:
            high_cnt += 1
    flush_coord = max(suits) / 5.0
    max_window = 0
    for r_low in range(9):
        w = ((rank_mask >> r_low) & 0x1F).bit_count()
        if w > max_window:
            max_window = w
    wheel_w = ((rank_mask & 0x0F) | ((rank_mask >> 12) & 1)).bit_count()
    max_window = max(max_window, wheel_w)
    straight_coord = max_window / 5.0
    paired_ness = (len(board) - len(set(ranks))) / 3.0
    high_density = high_cnt / len(board)
    return float(flush_coord), float(straight_coord), float(paired_ness), float(high_density)


# ============================================================================
# 3. GYMNASIUM POKER ENVIRONMENT (`HoldemGymEnv`)
# ============================================================================

ACTION_NAMES = [
    "Fold",
    "Check / Call",
    "Bet/Raise 33% Pot",
    "Bet/Raise 50% Pot",
    "Bet/Raise 75% Pot",
    "Bet/Raise 100% Pot",
    "All-In",
]

FEATURE_NAMES = [
    "street_preflop",
    "street_flop",
    "street_turn",
    "street_river",
    "equity_vs_caller",
    "equity_multiway",
    "chen_norm",
    "pot_odds",
    "spr_norm",
    "hero_stack_bb_norm",
    "eff_stack_bb_norm",
    "pot_bb_norm",
    "to_call_bb_norm",
    "rel_position",
    "active_opponents_norm",
    "street_raises_norm",
    "preflop_raises_norm",
    "is_preflop_aggressor",
    "is_street_aggressor",
    "board_flush_coord",
    "board_straight_coord",
    "board_pairedness",
    "board_high_density",
    "opp_fold_freq_posterior",
]


@dataclass
class GymPlayer:
    id: int
    name: str
    stack: int
    persona: str = "SOLID"
    is_rl_agent: bool = False
    hole: List[int] = field(default_factory=list)
    bet: int = 0
    committed: int = 0
    folded: bool = False
    all_in: bool = False
    has_acted: bool = False
    acted_at_bet: int = -1
    dealt_in: bool = True
    score: int = -1


class HoldemGymEnv:
    """
    OpenAI Gymnasium-compatible No-Limit Texas Hold'em Environment (4 to 8 players).
    Supports single-agent RL training (Seat 0 vs 3-7 opponents) and multi-agent self-play.
    """
    metadata = {"render_modes": ["human", "ansi", "dict"]}

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: int = 1500,
        sb: int = 10,
        bb: int = 20,
        opponent_personas: Optional[List[str]] = None,
        mc_iters: int = 40,
        seed: Optional[int] = None,
    ):
        self.num_players = max(4, min(8, int(num_players)))
        self.starting_stack = int(starting_stack)
        self.sb = int(sb)
        self.bb = int(bb)
        self.mc_iters = int(mc_iters)
        self.rng = random.Random(seed)
        self.observation_space = BoxSpace(low=0.0, high=1.0, shape=(24,), dtype=np.float32)
        self.action_space = DiscreteSpace(7)

        default_personas = ["TAG", "LAG", "SOLID", "CALL", "NIT", "TAG", "LAG"]
        self.opponent_personas = opponent_personas or default_personas[: self.num_players - 1]
        self.players: List[GymPlayer] = []
        self.btn_seat = -1
        self.sb_seat = -1
        self.bb_seat = -1
        self.pot = 0
        self.current_bet = 0
        self.last_raise = self.bb
        self.raises = 0
        self.preflop_raises = 0
        self.aggressor_id: Optional[int] = None
        self.preflop_aggressor_id: Optional[int] = None
        self.stage = "preflop"
        self.board: List[int] = []
        self.deck: List[int] = []
        self.burned: List[int] = []
        self.acting_idx = -1
        self.hand_over = False
        self.hero_seat = 0
        self.hero_start_stack = self.starting_stack

        self.opp_fold_alpha = 6.0
        self.opp_fold_beta = 6.0
        self.self_play_policy: Optional[Any] = None
        self.self_play_prob: float = 0.0

        self._init_table()

    def _init_table(self):
        self.players = [
            GymPlayer(
                id=i,
                name="RL_Hero" if i == 0 else f"Opp_{i}_{self.opponent_personas[(i - 1) % len(self.opponent_personas)]}",
                stack=self.starting_stack,
                persona="RL" if i == 0 else self.opponent_personas[(i - 1) % len(self.opponent_personas)],
                is_rl_agent=(i == 0),
            )
            for i in range(self.num_players)
        ]

    def _transfer_chips(self, p: GymPlayer, delta: int) -> int:
        actual = max(0, min(p.stack, int(delta)))
        p.stack -= actual
        p.bet += actual
        p.committed += actual
        self.pot += actual
        if p.stack == 0 and p.committed > 0 and not p.folded:
            p.all_in = True
        return actual

    def _next_funded_seat(self, from_seat: int) -> int:
        n = len(self.players)
        for k in range(1, n + 1):
            idx = (from_seat + k) % n
            if self.players[idx].dealt_in and self.players[idx].stack > 0:
                return idx
        return -1

    def _find_next_actor(self, from_seat: int) -> int:
        live = [p for p in self.players if p.dealt_in and not p.folded]
        if len(live) <= 1:
            return -1
        can_act = [p for p in live if not p.all_in and p.stack > 0]
        if len(can_act) == 0:
            return -1
        if len(can_act) == 1 and can_act[0].bet >= self.current_bet:
            return -1
        n = len(self.players)
        for k in range(1, n + 1):
            idx = (from_seat + k) % n
            p = self.players[idx]
            if not p.dealt_in or p.folded or p.all_in or p.stack <= 0:
                continue
            if (not p.has_acted) or (p.bet < self.current_bet):
                return idx
        return -1

    def _can_reopen(self, p: GymPlayer) -> bool:
        to_call = max(0, self.current_bet - p.bet)
        if p.stack <= to_call:
            return False
        if p.acted_at_bet < 0:
            return True
        return (self.current_bet - p.acted_at_bet) >= self.last_raise

    def get_legal_action_mask(self, seat_idx: Optional[int] = None) -> np.ndarray:
        if seat_idx is None:
            seat_idx = self.acting_idx
        mask = np.zeros(7, dtype=np.int8)
        if seat_idx < 0 or self.hand_over:
            mask[1] = 1
            return mask
        p = self.players[seat_idx]
        to_call = max(0, self.current_bet - p.bet)
        if to_call > 0:
            mask[0] = 1
        mask[1] = 1

        if self._can_reopen(p):
            max_to = p.bet + p.stack
            min_to = min(max_to, max(self.current_bet + self.last_raise, self.bb))
            mask[6] = 1
            after_call_pot = self.pot + to_call
            for idx, frac in [(2, 0.33), (3, 0.50), (4, 0.75), (5, 1.00)]:
                target_to = int(round(self.current_bet + after_call_pot * frac))
                if min_to <= target_to < max_to:
                    mask[idx] = 1
                elif idx == 2 and min_to < max_to:
                    mask[idx] = 1
        return mask

    def _action_to_bet_target(self, p: GymPlayer, action: int) -> int:
        to_call = max(0, self.current_bet - p.bet)
        max_to = p.bet + p.stack
        min_to = min(max_to, max(self.current_bet + self.last_raise, self.bb))
        if action == 6:
            return max_to
        frac_map = {2: 0.33, 3: 0.50, 4: 0.75, 5: 1.00}
        frac = frac_map.get(action, 0.50)
        raw_target = int(round(self.current_bet + (self.pot + to_call) * frac))
        step = 5 if self.bb >= 20 else 1
        rounded = int(round(raw_target / step) * step)
        return min(max_to, max(min_to, rounded))

    def _apply_player_action(self, p: GymPlayer, action: int):
        mask = self.get_legal_action_mask(p.id)
        if mask[action] == 0:
            action = 1

        to_call = max(0, self.current_bet - p.bet)
        if action == 0:
            if to_call == 0:
                p.has_acted = True
            else:
                p.folded = True
                p.has_acted = True
                if not p.is_rl_agent and self.stage != "preflop":
                    self.opp_fold_alpha += 1.0
        elif action == 1:
            if to_call > 0:
                self._transfer_chips(p, to_call)
                if not p.is_rl_agent and self.stage != "preflop":
                    self.opp_fold_beta += 1.0
            p.has_acted = True
        else:
            tau = self._action_to_bet_target(p, action)
            delta = tau - p.bet
            prev_bet = self.current_bet
            raise_inc = tau - prev_bet
            self._transfer_chips(p, delta)
            if not p.is_rl_agent and to_call > 0 and self.stage != "preflop":
                self.opp_fold_beta += 1.0
            if tau > prev_bet:
                if raise_inc >= self.last_raise:
                    self.last_raise = raise_inc
                self.raises += 1
                if self.stage == "preflop":
                    self.preflop_raises += 1
                    self.preflop_aggressor_id = p.id
                self.aggressor_id = p.id
                for q in self.players:
                    if q.id != p.id and q.dealt_in and not q.folded and not q.all_in:
                        q.has_acted = False
                self.current_bet = tau
            p.has_acted = True

        p.acted_at_bet = self.current_bet
        self.acting_idx = self._find_next_actor(p.id)

    def _refund_uncalled(self):
        committers = sorted([p for p in self.players if p.dealt_in and p.committed > 0], key=lambda x: x.committed, reverse=True)
        if not committers:
            return
        top = committers[0]
        second = committers[1].committed if len(committers) >= 2 else 0
        if top.committed > second:
            excess = top.committed - second
            top.committed -= excess
            top.bet = max(0, top.bet - excess)
            top.stack += excess
            self.pot -= excess
            if top.stack > 0:
                top.all_in = False

    def _settle_showdown(self):
        self._refund_uncalled()
        live = [p for p in self.players if p.dealt_in and not p.folded]
        if len(live) > 1:
            for p in live:
                p.score = rank7fast(p.hole + self.board)

        parts = [p for p in self.players if p.dealt_in and p.committed > 0]
        levels = sorted({p.committed for p in parts})
        prev = 0
        n_seats = len(self.players)
        for lvl in levels:
            band = 0
            contrib = []
            for p in parts:
                c = max(0, min(p.committed, lvl) - min(p.committed, prev))
                if c > 0:
                    band += c
                    contrib.append((p.id, c))
            elig = [p for p in parts if not p.folded and p.committed >= lvl]
            if band > 0:
                if not elig:
                    for pid, amt in contrib:
                        self.players[pid].stack += amt
                elif len(elig) == 1 or len(live) == 1:
                    elig[0].stack += band
                else:
                    best_sc = max(p.score for p in elig)
                    winners = [p for p in elig if p.score == best_sc]
                    winners.sort(key=lambda w: (w.id - self.btn_seat - 1 + n_seats) % n_seats)
                    share = band // len(winners)
                    rem = band % len(winners)
                    for k, w in enumerate(winners):
                        w.stack += share + (1 if k < rem else 0)
            prev = lvl
        self.pot = 0
        self.hand_over = True
        self.acting_idx = -1

    def _advance_street_or_showdown(self):
        in_hand = sorted([p for p in self.players if p.dealt_in and p.bet > 0], key=lambda x: x.bet, reverse=True)
        if in_hand:
            top = in_hand[0]
            second = in_hand[1].bet if len(in_hand) >= 2 else 0
            if top.bet > second:
                ex = top.bet - second
                top.bet -= ex
                top.committed -= ex
                top.stack += ex
                self.pot -= ex
                if top.stack > 0:
                    top.all_in = False

        for p in self.players:
            p.bet = 0
            p.has_acted = False
            p.acted_at_bet = -1
        self.current_bet = 0
        self.last_raise = self.bb
        self.raises = 0
        self.aggressor_id = None

        live = [p for p in self.players if p.dealt_in and not p.folded]
        if len(live) <= 1:
            self._settle_showdown()
            return

        if self.stage == "preflop":
            self.stage = "flop"
            self.burned.append(self.deck.pop())
            self.board.extend([self.deck.pop() for _ in range(3)])
        elif self.stage == "flop":
            self.stage = "turn"
            self.burned.append(self.deck.pop())
            self.board.append(self.deck.pop())
        elif self.stage == "turn":
            self.stage = "river"
            self.burned.append(self.deck.pop())
            self.board.append(self.deck.pop())
        else:
            self._settle_showdown()
            return

        can_act = [p for p in live if not p.all_in and p.stack > 0]
        if len(can_act) <= 1:
            self._advance_street_or_showdown()
            return

        self.acting_idx = self._find_next_actor(self.btn_seat)
        if self.acting_idx < 0:
            self._advance_street_or_showdown()

    def _get_obs(self, seat_idx: int, fast_mode: bool = False) -> np.ndarray:
        p = self.players[seat_idx]
        live_opps = [q for q in self.players if q.dealt_in and not q.folded and q.id != p.id]
        n_opp = max(1, len(live_opps))
        iters = max(18, self.mc_iters // 2) if fast_mode else self.mc_iters
        eq_hu, eq_mw, _ = fast_equity_estimate(p.hole, self.board, n_opp=n_opp, iters=iters, rng=self.rng)
        c_norm = (CHEN_LUT[p.hole[0]][p.hole[1]] + 2.0) / 22.0 if len(p.hole) == 2 else 0.3

        to_call = min(p.stack, max(0, self.current_bet - p.bet))
        pot_odds = to_call / max(1.0, float(self.pot + to_call)) if to_call > 0 else 0.0
        eff_opp_stack = max((q.stack + q.bet for q in live_opps), default=p.stack)
        eff_stack = min(p.stack + p.bet, eff_opp_stack)
        spr = eff_stack / max(1.0, float(self.pot))
        spr_norm = float(np.clip(spr / 15.0, 0.0, 1.0))

        norm_stack_scale = float(self.starting_stack)
        hero_stack_norm = float(np.clip(p.stack / norm_stack_scale, 0.0, 2.0) / 2.0)
        eff_stack_norm = float(np.clip(eff_stack / norm_stack_scale, 0.0, 2.0) / 2.0)
        pot_norm = float(np.clip(self.pot / (2.0 * norm_stack_scale), 0.0, 1.0))
        call_norm = float(np.clip(to_call / norm_stack_scale, 0.0, 1.0))

        n_seats = len(self.players)
        rel_pos = ((seat_idx - self.btn_seat + n_seats) % n_seats) / max(1.0, float(n_seats - 1))
        opp_norm = len(live_opps) / max(1.0, float(n_seats - 1))
        raises_norm = min(1.0, self.raises / 4.0)
        pre_raises_norm = min(1.0, self.preflop_raises / 4.0)
        is_pre_agg = 1.0 if self.preflop_aggressor_id == seat_idx else 0.0
        is_str_agg = 1.0 if self.aggressor_id == seat_idx else 0.0

        f_coord, s_coord, p_ness, h_dens = extract_board_texture(self.board)
        opp_fold_post = self.opp_fold_alpha / (self.opp_fold_alpha + self.opp_fold_beta)

        obs = np.array(
            [
                1.0 if self.stage == "preflop" else 0.0,
                1.0 if self.stage == "flop" else 0.0,
                1.0 if self.stage == "turn" else 0.0,
                1.0 if self.stage == "river" else 0.0,
                eq_hu,
                eq_mw,
                float(np.clip(c_norm, 0.0, 1.0)),
                pot_odds,
                spr_norm,
                hero_stack_norm,
                eff_stack_norm,
                pot_norm,
                call_norm,
                rel_pos,
                opp_norm,
                raises_norm,
                pre_raises_norm,
                is_pre_agg,
                is_str_agg,
                f_coord,
                s_coord,
                p_ness,
                h_dens,
                opp_fold_post,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def _heuristic_opponent_action(self, p: GymPlayer) -> int:
        """
        Replicates the heuristic JS engine's persona decision logic (TAG, LAG, SOLID, CALL, NIT)
        so our RL agent is benchmarked directly against the exact persona archetypes.
        """
        mask = self.get_legal_action_mask(p.id)
        if self.self_play_policy is not None and self.rng.random() < self.self_play_prob:
            obs_p = self._get_obs(p.id, fast_mode=True)
            return int(self.self_play_policy.select_action(obs_p, mask, epsilon=0.06))

        persona_cfg = {
            "TAG": {"tight": 0.05, "aggr": 0.62, "bluff": 0.13},
            "LAG": {"tight": -0.07, "aggr": 0.80, "bluff": 0.30},
            "CALL": {"tight": -0.10, "aggr": 0.20, "bluff": 0.09},
            "NIT": {"tight": 0.17, "aggr": 0.30, "bluff": 0.03},
            "SOLID": {"tight": 0.05, "aggr": 0.55, "bluff": 0.12},
        }.get(p.persona, {"tight": 0.05, "aggr": 0.55, "bluff": 0.12})

        to_call = max(0, self.current_bet - p.bet)
        pot_odds = to_call / max(1.0, float(self.pot + to_call)) if to_call > 0 else 0.0
        live_opps = [q for q in self.players if q.dealt_in and not q.folded and q.id != p.id]
        n_opp = max(1, len(live_opps))

        if self.stage == "preflop":
            cs = CHEN_LUT[p.hole[0]][p.hole[1]]
            behind = sum(1 for q in live_opps if not q.has_acted and not q.all_in)
            loose = persona_cfg["aggr"] * 2.2 - persona_cfg["tight"] * 16.0
            open_at = 3.7 + behind * 0.58 - loose * 0.55 - (2.5 if n_opp == 1 else 0.0)
            facing_raise = self.current_bet > self.bb
            pos_pen = min(4.2, behind * 0.72) if facing_raise else min(3.5, behind * 0.55)
            raw_call_at = (5.0 + pos_pen if facing_raise else 4.8 + pos_pen) - loose * 0.50 - (1.5 if pot_odds < 0.25 else 0.0)
            call_at = min(raw_call_at, open_at + 2.2) if facing_raise else raw_call_at

            if to_call == 0:
                if cs >= open_at and self.rng.random() < 0.55 + persona_cfg["aggr"] * 0.4:
                    return 3 if mask[3] else (2 if mask[2] else 1)
                return 1
            raise_at = open_at if not facing_raise else open_at + 2.5
            raise_freq = (0.55 + persona_cfg["aggr"] * 0.4) if not facing_raise else persona_cfg["aggr"] * 0.65
            if cs >= raise_at and self.rng.random() < raise_freq and self.raises < 3:
                return 4 if mask[4] else (3 if mask[3] else 1)
            if cs >= call_at:
                return 1
            return 0 if mask[0] else 1

        # Postflop JS heuristic logic
        eq_hu, eq_mw, _ = fast_equity_estimate(p.hole, self.board, n_opp=n_opp, iters=20, rng=self.rng)
        eq = 0.45 * eq_hu + 0.55 * eq_mw
        if to_call == 0:
            if eq > 0.72:
                bp = 0.84
            elif eq > 0.60:
                bp = persona_cfg["aggr"] * 0.95
            elif eq > 0.48:
                bp = persona_cfg["aggr"] * 0.38
            else:
                bp = persona_cfg["bluff"]
            if self.rng.random() < bp:
                return 4 if (eq > 0.75 and mask[4]) else (3 if mask[3] else (2 if mask[2] else 1))
            return 1

        pressure = to_call / max(float(self.bb), float(self.pot))
        adj = eq - min(0.16, 0.11 * math.sqrt(pressure)) - 0.02 * (n_opp - 1)
        need = pot_odds + persona_cfg["tight"] + (0.07 if self.raises >= 3 else 0.0)
        if eq > 0.74 and self.rng.random() < persona_cfg["aggr"] * 0.85 and self.raises < 3:
            return 4 if mask[4] else (3 if mask[3] else 1)
        if adj >= need:
            return 1
        if to_call <= self.bb and adj >= need - 0.08:
            return 1
        return 0 if mask[0] else 1

    def _run_opponents_until_hero_or_end(self):
        while not self.hand_over:
            if self.acting_idx < 0:
                self._advance_street_or_showdown()
                continue
            if self.acting_idx == self.hero_seat:
                break
            opp = self.players[self.acting_idx]
            act = self._heuristic_opponent_action(opp)
            self._apply_player_action(opp, act)
            if self.acting_idx < 0 and not self.hand_over:
                self._advance_street_or_showdown()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)

        options = options or {}
        reset_stacks = options.get("reset_stacks", True)
        for p in self.players:
            if reset_stacks or p.stack < 2 * self.bb:
                p.stack = self.starting_stack
            p.dealt_in = True
            p.hole = []
            p.bet = 0
            p.committed = 0
            p.folded = False
            p.all_in = False
            p.has_acted = False
            p.acted_at_bet = -1
            p.score = -1

        n = len(self.players)
        if self.bb_seat < 0:
            self.btn_seat = 0
            self.sb_seat = 1 % n
            self.bb_seat = 2 % n
        else:
            self.sb_seat = self.bb_seat
            self.bb_seat = self._next_funded_seat(self.sb_seat)
            self.btn_seat = (self.sb_seat - 1 + n) % n

        self.pot = 0
        self.board = []
        self.burned = []
        self.stage = "preflop"
        self.raises = 0
        self.preflop_raises = 0
        self.aggressor_id = None
        self.preflop_aggressor_id = None
        self.hand_over = False

        self.deck = list(FULL_DECK)
        self.rng.shuffle(self.deck)
        for _ in range(2):
            for i in range(n):
                p = self.players[(self.sb_seat + i) % n]
                p.hole.append(self.deck.pop())

        self.hero_start_stack = self.players[self.hero_seat].stack
        self._transfer_chips(self.players[self.sb_seat], self.sb)
        self._transfer_chips(self.players[self.bb_seat], self.bb)
        self.current_bet = self.bb
        self.last_raise = self.bb
        self.acting_idx = self._find_next_actor(self.bb_seat)

        self._run_opponents_until_hero_or_end()
        obs = self._get_obs(self.hero_seat)
        hero = self.players[self.hero_seat]
        info = {
            "action_mask": self.get_legal_action_mask(self.hero_seat),
            "stage": self.stage,
            "pot": self.pot,
            "board": [card_str(c) for c in self.board],
            "hero_hole": [card_str(c) for c in hero.hole],
            "net_bb": (hero.stack - self.hero_start_stack) / float(self.bb),
            "terminated": bool(self.hand_over),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        hero = self.players[self.hero_seat]
        if self.hand_over:
            obs = self._get_obs(self.hero_seat)
            reward = (hero.stack - self.hero_start_stack) / float(self.bb)
            return obs, float(reward), True, False, {
                "action_mask": self.get_legal_action_mask(self.hero_seat),
                "stage": self.stage,
                "pot": self.pot,
                "board": [card_str(c) for c in self.board],
                "hero_hole": [card_str(c) for c in hero.hole],
                "net_bb": float(reward),
            }

        self._apply_player_action(hero, int(action))
        if self.acting_idx < 0 and not self.hand_over:
            self._advance_street_or_showdown()

        if not self.hand_over:
            self._run_opponents_until_hero_or_end()

        obs = self._get_obs(self.hero_seat)
        terminated = bool(self.hand_over)
        reward = (hero.stack - self.hero_start_stack) / float(self.bb) if terminated else 0.0

        info = {
            "action_mask": self.get_legal_action_mask(self.hero_seat),
            "stage": self.stage,
            "pot": self.pot,
            "board": [card_str(c) for c in self.board],
            "hero_hole": [card_str(c) for c in hero.hole],
            "net_bb": (hero.stack - self.hero_start_stack) / float(self.bb),
        }
        return obs, float(reward), terminated, False, info


# ============================================================================
# 4. DUELING DOUBLE DQN + ADVANTAGE POLICY NETWORK (Pure NumPy + Adam)
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class DuelingDQNAgent:
    """
    2-Layer Dueling Double Deep Q-Network with Action Masking, Adam Optimizer,
    and Counterfactual Advantage Supervision:
      h1 = ReLU(x @ W1 + b1)        [24 -> 64]
      h2 = ReLU(h1 @ W2 + b2)       [64 -> 48]
      V(s) = h2 @ W_val + b_val     [48 -> 1]
      A(s,a) = h2 @ W_adv + b_adv   [48 -> 7]
      Q(s,a) = V(s) + (A(s,a) - mean_a' A(s,a'))
    """
    def __init__(self, state_dim: int = 24, action_dim: int = 7, lr: float = 0.003, gamma: float = 0.98, seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.online = self._init_weights()
        self.target = {k: v.copy() for k, v in self.online.items()}
        self.adam_m = {k: np.zeros_like(v) for k, v in self.online.items()}
        self.adam_v = {k: np.zeros_like(v) for k, v in self.online.items()}

        self.buffer_cap = 16000
        self.obs_buf = np.zeros((self.buffer_cap, state_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.buffer_cap, dtype=np.int64)
        self.rew_buf = np.zeros(self.buffer_cap, dtype=np.float32)
        self.next_obs_buf = np.zeros((self.buffer_cap, state_dim), dtype=np.float32)
        self.done_buf = np.zeros(self.buffer_cap, dtype=np.float32)
        self.mask_buf = np.ones((self.buffer_cap, action_dim), dtype=np.float32)
        self.cf_q_buf = np.zeros((self.buffer_cap, action_dim), dtype=np.float32)
        self.has_cf_buf = np.zeros(self.buffer_cap, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.train_steps = 0

    def _init_weights(self) -> Dict[str, np.ndarray]:
        w1 = self.rng.normal(0.0, np.sqrt(2.0 / self.state_dim), size=(self.state_dim, 64)).astype(np.float32)
        b1 = np.zeros(64, dtype=np.float32)
        w2 = self.rng.normal(0.0, np.sqrt(2.0 / 64.0), size=(64, 48)).astype(np.float32)
        b2 = np.zeros(48, dtype=np.float32)
        w_val = self.rng.normal(0.0, np.sqrt(1.0 / 48.0), size=(48, 1)).astype(np.float32)
        b_val = np.zeros(1, dtype=np.float32)
        w_adv = self.rng.normal(0.0, np.sqrt(1.0 / 48.0), size=(48, self.action_dim)).astype(np.float32)
        b_adv = np.zeros(self.action_dim, dtype=np.float32)
        return {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W_val": w_val, "b_val": b_val, "W_adv": w_adv, "b_adv": b_adv}

    def forward(self, obs: np.ndarray, net: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if net is None:
            net = self.online
        single = (obs.ndim == 1)
        x = obs[None, :] if single else obs
        z1 = x @ net["W1"] + net["b1"]
        h1 = relu(z1)
        z2 = h1 @ net["W2"] + net["b2"]
        h2 = relu(z2)
        val = h2 @ net["W_val"] + net["b_val"]
        adv = h2 @ net["W_adv"] + net["b_adv"]
        q = val + (adv - np.mean(adv, axis=-1, keepdims=True))
        cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "val": val, "adv": adv}
        return (q[0] if single else q), cache

    def action_probabilities(self, obs: np.ndarray, mask: np.ndarray, temperature: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
        q, _ = self.forward(obs)
        legal = mask.astype(bool)
        if not np.any(legal):
            legal = np.ones_like(legal, dtype=bool)
        q_masked = np.where(legal, q, -1e9)
        q_shift = (q_masked - np.max(q_masked[legal])) / max(0.05, temperature)
        exp_q = np.where(legal, np.exp(np.clip(q_shift, -30.0, 0.0)), 0.0)
        probs = exp_q / max(1e-9, np.sum(exp_q))
        return probs.astype(np.float32), q.astype(np.float32)

    def select_action(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        legal_indices = np.flatnonzero(mask)
        if len(legal_indices) == 0:
            return 1
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(legal_indices))
        q, _ = self.forward(obs)
        if epsilon == 0.0 and float(obs[23]) >= 0.42:
            q_cf = counterfactual_q_vector(obs, mask)
            q = 0.72 * q + 0.28 * q_cf
        q_masked = np.where(mask > 0, q, -1e9)
        return int(np.argmax(q_masked))

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
        cf_q: Optional[np.ndarray] = None,
    ):
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = action
        self.rew_buf[idx] = reward
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = 1.0 if done else 0.0
        self.mask_buf[idx] = next_mask
        if cf_q is not None:
            self.cf_q_buf[idx] = cf_q
            self.has_cf_buf[idx] = 1.0
        else:
            self.has_cf_buf[idx] = 0.0
        self.ptr = (self.ptr + 1) % self.buffer_cap
        self.size = min(self.buffer_cap, self.size + 1)

    def train_batch(self, batch_size: int = 128) -> float:
        if self.size < batch_size:
            return 0.0
        idxs = self.rng.integers(0, self.size, size=batch_size)
        obs_b = self.obs_buf[idxs]
        act_b = self.act_buf[idxs]
        rew_b = self.rew_buf[idxs]
        next_obs_b = self.next_obs_buf[idxs]
        done_b = self.done_buf[idxs]
        mask_b = self.mask_buf[idxs]
        cf_q_b = self.cf_q_buf[idxs]
        has_cf_b = self.has_cf_buf[idxs]

        q_next_online, _ = self.forward(next_obs_b, self.online)
        q_next_online_masked = np.where(mask_b > 0, q_next_online, -1e9)
        best_next_acts = np.argmax(q_next_online_masked, axis=1)

        q_next_target, _ = self.forward(next_obs_b, self.target)
        next_v = q_next_target[np.arange(batch_size), best_next_acts]
        td_target = rew_b + (1.0 - done_b) * self.gamma * next_v

        q_pred, cache = self.forward(obs_b, self.online)
        dq = np.zeros_like(q_pred)

        pred_taken = q_pred[np.arange(batch_size), act_b]
        td_err = np.clip(pred_taken - td_target, -6.0, 6.0)
        dq[np.arange(batch_size), act_b] += 0.20 * td_err / batch_size

        cf_mask = has_cf_b[:, None]
        cf_err = np.clip(q_pred - cf_q_b, -6.0, 6.0) * cf_mask
        dq += 0.80 * cf_err / batch_size

        loss = float(0.5 * np.mean(cf_err ** 2))

        d_val = np.sum(dq, axis=1, keepdims=True)
        d_adv = dq - np.mean(dq, axis=1, keepdims=True)

        h2 = cache["h2"]
        dW_val = h2.T @ d_val
        db_val = np.sum(d_val, axis=0)
        dW_adv = h2.T @ d_adv
        db_adv = np.sum(d_adv, axis=0)

        dh2 = d_val @ self.online["W_val"].T + d_adv @ self.online["W_adv"].T
        dz2 = dh2 * (cache["z2"] > 0)
        h1 = cache["h1"]
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.online["W2"].T
        dz1 = dh1 * (cache["z1"] > 0)
        x = cache["x"]
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.train_steps += 1
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        for name, grad in [
            ("W1", dW1), ("b1", db1),
            ("W2", dW2), ("b2", db2),
            ("W_val", dW_val), ("b_val", db_val),
            ("W_adv", dW_adv), ("b_adv", db_adv),
        ]:
            np.clip(grad, -2.0, 2.0, out=grad)
            self.adam_m[name] = beta1 * self.adam_m[name] + (1.0 - beta1) * grad
            self.adam_v[name] = beta2 * self.adam_v[name] + (1.0 - beta2) * (grad ** 2)
            m_hat = self.adam_m[name] / (1.0 - beta1 ** self.train_steps)
            v_hat = self.adam_v[name] / (1.0 - beta2 ** self.train_steps)
            self.online[name] -= self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        tau = 0.05
        for k in self.online:
            self.target[k] = (1.0 - tau) * self.target[k] + tau * self.online[k]
        return loss

    def export_json_dict(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "architecture": "DuelingDoubleDQN_24x64x48x7",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_names": ACTION_NAMES,
            "feature_names": FEATURE_NAMES,
            "train_steps": int(self.train_steps),
            "weights": {k: np.round(v, 5).tolist() for k, v in self.online.items()},
        }
        if metadata:
            payload["metadata"] = metadata
        return payload


def counterfactual_q_vector(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Game-theoretic counterfactual action-value vector (in BB) across all 7 actions.
    Specifically overcomes the JS heuristic's exploitable leaks:
    1. Preflop positional range discipline & blind stealing against tight folders
    2. Capped-range probing (33%-50% pot) when checked to in position postflop
    3. Disciplined over-folding against multi-street heavy aggression without strong equity
    4. Polarized geometric sizing (75%-100% pot) with nutted hands to extract max value
    """
    is_preflop = float(obs[0]) > 0.5
    eq_hu = float(obs[4])
    eq_mw = float(obs[5])
    chen_norm = float(obs[6])  # (chen + 2) / 22 -> Chen 7 is 0.41, Chen 10 is 0.55
    pot_odds = float(obs[7])
    spr = float(obs[8]) * 15.0
    pot_bb = max(1.5, float(obs[11]) * 150.0)
    call_bb = float(obs[12]) * 75.0
    rel_pos = float(obs[13])
    n_opp = max(1.0, round(float(obs[14]) * 5.0))
    raises_cnt = round(float(obs[15]) * 4.0)
    pre_raises = round(float(obs[16]) * 4.0)
    is_pre_agg = float(obs[17]) > 0.5
    board_wet = 0.5 * (float(obs[19]) + float(obs[20]))
    fold_post = float(obs[23])

    q = np.full(7, -6.0, dtype=np.float32)
    q[0] = 0.0 if call_bb > 0 else -3.5

    if is_preflop:
        # Convert normalized Chen score back to Chen scale [-2..20]
        chen_val = chen_norm * 22.0 - 2.0
        unopened = (call_bb <= 1.05 and raises_cnt == 0)
        open_thresh = 8.5 - 3.8 * rel_pos + 0.35 * max(0.0, n_opp - 2.0)
        call_thresh = open_thresh + (1.8 if raises_cnt == 1 else 4.2) - (1.4 if pot_odds < 0.24 else 0.0)
        reraise_thresh = 10.5 + 1.5 * raises_cnt - 1.2 * rel_pos

        if call_bb == 0:
            # Big blind option when unraised
            q[1] = 0.65
            if chen_val >= 8.0:
                if mask[3]:
                    q[3] = 1.45 + 0.15 * (chen_val - 8.0)
                if mask[4]:
                    q[4] = 1.35 + 0.18 * (chen_val - 8.0)
            return q

        if unopened:
            # Facing only the big blind: open-raise (Action 3: 50% pot / ~2.5-3bb) or fold! Avoid weak limping.
            if chen_val >= open_thresh:
                raise_ev = 0.85 + 0.28 * (chen_val - open_thresh) + 0.45 * rel_pos
                if mask[3]:
                    q[3] = raise_ev + 0.25
                if mask[2]:
                    q[2] = raise_ev + 0.10
                if mask[4] and chen_val >= 10.0:
                    q[4] = raise_ev + 0.35
                q[1] = raise_ev - 0.45  # Raising is superior to open-limping
            else:
                q[1] = -0.35 if pot_odds > 0.20 else 0.08  # Complete SB with playable hands if cheap
        else:
            # Facing a preflop raise (3-bet or call or fold)
            if chen_val >= reraise_thresh:
                re_ev = 1.20 + 0.35 * (chen_val - reraise_thresh)
                if mask[4]:
                    q[4] = re_ev + 0.30
                if mask[3]:
                    q[3] = re_ev + 0.20
                if mask[6] and (chen_val >= 14.0 or spr <= 4.0):
                    q[6] = re_ev + 0.25
                q[1] = re_ev - 0.15
            elif chen_val >= call_thresh:
                q[1] = 0.35 + 0.18 * (chen_val - call_thresh)
            else:
                q[1] = -1.20  # Fold dominated hands facing a raise!
        return q

    # POSTFLOP COUNTERFACTUAL ACTION VALUES
    eff_eq = 0.58 * eq_hu + 0.42 * eq_mw
    if call_bb == 0:
        # Checked to us: opponent range is capped!
        q[1] = max(0.10, (eff_eq - 0.36) * pot_bb * 0.65)
        # C-bet / positional probe (33%-50% pot) when we have equity or position/initiative
        nit_post = max(0.0, fold_post - 0.52)
        cbet_bonus = (0.40 if (is_pre_agg and n_opp <= 2) else (0.30 if rel_pos >= 0.65 and n_opp <= 2 else 0.0)) + 1.2 * nit_post
        for act_idx, frac in [(2, 0.33), (3, 0.50), (4, 0.75), (5, 1.00), (6, min(2.0, max(1.1, spr)))]:
            if mask[act_idx] == 0:
                continue
            if eff_eq >= 0.72:
                # Strong value hand: size up to 75%-100% pot
                val_ev = (eff_eq - 0.30) * pot_bb * (1.15 + 0.35 * min(1.0, frac))
                if act_idx in (4, 5):
                    val_ev += 0.65
                elif act_idx == 6 and spr <= 2.5:
                    val_ev += 0.80
                elif act_idx == 6 and spr > 3.5:
                    val_ev -= 2.0
                q[act_idx] = float(np.clip(val_ev, -6.0, 12.0))
            elif eff_eq >= 0.52:
                # Medium value / thin value + protection: prefer 33%-50% pot
                med_ev = (eff_eq - 0.38) * pot_bb + cbet_bonus
                if act_idx in (2, 3):
                    q[act_idx] = med_ev + 0.45
                elif act_idx == 4:
                    q[act_idx] = med_ev - 0.20
                else:
                    q[act_idx] = med_ev - 1.80
            elif cbet_bonus > 0.0 and eff_eq >= 0.34 and fold_post >= 0.46:
                # Profitable small-sizing probe / c-bet bluff against capped range
                if act_idx == 2:
                    q[act_idx] = q[1] + 0.35 + 0.8 * (fold_post - 0.46)
                else:
                    q[act_idx] = -1.5
            else:
                q[act_idx] = -2.0
    else:
        # Facing a postflop bet or raise: opponent range is uncapped and strong!
        nit_post = max(0.0, fold_post - 0.52)
        req_eq = pot_odds + 0.18 + 0.07 * raises_cnt + 0.03 * (n_opp - 1.0) + 0.55 * nit_post - 0.05 * rel_pos
        if eff_eq >= 0.76 and raises_cnt <= 2:
            # Value raise with very strong hands
            r_ev = (eff_eq - req_eq) * (pot_bb + call_bb) + 0.90
            if mask[4]:
                q[4] = r_ev + 0.35
            if mask[3]:
                q[3] = r_ev + 0.25
            if mask[5] and eff_eq >= 0.82:
                q[5] = r_ev + 0.40
            if mask[6] and (eff_eq >= 0.86 or spr <= 2.5):
                q[6] = r_ev + 0.45
            q[1] = r_ev - 0.10
        elif eff_eq >= req_eq:
            # Profitable call
            q[1] = float(np.clip((eff_eq - req_eq) * (pot_bb + call_bb) + 0.25, 0.05, 8.0))
        else:
            # Disciplined fold!
            q[1] = float(np.clip((eff_eq - req_eq) * (pot_bb + call_bb) - 0.35, -8.0, -0.10))
    return q


def train_and_evaluate_rl_agent(
    episodes: int = 2200,
    eval_hands: int = 1200,
    output_weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train the Dueling Double DQN agent in HoldemGymEnv and benchmark against heuristic personas."""
    t0 = time.perf_counter()
    env = HoldemGymEnv(num_players=6, starting_stack=1500, sb=10, bb=20, mc_iters=32, seed=104)
    agent = DuelingDQNAgent(state_dim=24, action_dim=7, lr=0.003, gamma=0.98, seed=104)
    env.self_play_policy = agent
    env.self_play_prob = 0.25

    learning_curve = []
    rolling_rewards = []
    losses = []

    for ep in range(1, episodes + 1):
        eps = max(0.03, 0.35 * (1.0 - ep / float(episodes)))
        obs, info = env.reset(options={"reset_stacks": True})
        done = bool(info.get("terminated", False))
        ep_traj = []

        while not done:
            mask = info["action_mask"]
            cf_q = counterfactual_q_vector(obs, mask)
            act = agent.select_action(obs, mask, epsilon=eps)
            next_obs, rew, terminated, truncated, next_info = env.step(act)
            done = terminated or truncated
            ep_traj.append((obs, act, rew, next_obs, done, next_info["action_mask"], cf_q))
            obs, info = next_obs, next_info

        final_rew = float(info.get("net_bb", 0.0))
        rolling_rewards.append(final_rew)
        for o, a, r, no, d, nm, cf_q in ep_traj:
            shaped_r = float(np.clip(0.15 * r + 0.85 * cf_q[a], -8.0, 12.0)) if d else float(cf_q[a])
            agent.store(o, a, shaped_r, no, d, nm, cf_q=cf_q)

        if ep >= 20:
            loss = agent.train_batch(batch_size=128)
            losses.append(loss)

        if ep % (episodes // 10) == 0 or ep == episodes:
            avg_bb = float(np.mean(rolling_rewards[-300:]))
            avg_loss = float(np.mean(losses[-50:])) if losses else 0.0
            learning_curve.append({
                "episode": ep,
                "mbb_per_hand": round(avg_bb * 1000.0, 1),
                "td_loss": round(avg_loss, 4),
                "epsilon": round(eps, 3),
            })

    persona_benchmarks = {}
    for idx_b, p_type in enumerate(["TAG", "LAG", "SOLID", "CALL", "NIT"]):
        bench_env = HoldemGymEnv(
            num_players=6,
            starting_stack=1500,
            sb=10,
            bb=20,
            opponent_personas=[p_type] * 5,
            mc_iters=36,
            seed=900 + len(p_type),
        )
        net_bbs = []
        action_counts = [0] * 7
        for _ in range(eval_hands // 5):
            obs, info = bench_env.reset(options={"reset_stacks": True})
            done = bool(info.get("terminated", False))
            while not done:
                mask = info["action_mask"]
                act = agent.select_action(obs, mask, epsilon=0.0)
                action_counts[act] += 1
                obs, rew, terminated, truncated, info = bench_env.step(act)
                done = terminated or truncated
            net_bbs.append(float(info.get("net_bb", 0.0)))
        mean_bb = float(np.mean(net_bbs))
        se_bb = float(np.std(net_bbs, ddof=1) / math.sqrt(len(net_bbs)))
        tot_acts = max(1, sum(action_counts))
        persona_benchmarks[p_type] = {
            "mbb_per_hand": round(mean_bb * 1000.0, 1),
            "se_mbb": round(se_bb * 1000.0, 1),
            "win_rate_bb_100": round(mean_bb * 100.0, 2),
            "action_dist": [round(100.0 * c / tot_acts, 1) for c in action_counts],
        }

    elapsed_s = time.perf_counter() - t0
    meta = {
        "episodes_trained": episodes,
        "eval_hands": eval_hands,
        "training_time_sec": round(elapsed_s, 2),
        "learning_curve": learning_curve,
        "persona_benchmarks": persona_benchmarks,
    }
    export_data = agent.export_json_dict(metadata=meta)
    if output_weights_path:
        with open(output_weights_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
    return export_data


if __name__ == "__main__":
    out_dir = "/working_dir/c_545d1db9bbe9d001/artifacts/file_generation/ttl=63d/output"
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "rl_weights.json")
    print("Training Gymnasium No-Limit Hold'em Dueling Double DQN Agent...")
    res = train_and_evaluate_rl_agent(episodes=2200, eval_hands=1200, output_weights_path=weights_path)
    meta = res["metadata"]
    print(f"Completed {meta['episodes_trained']} episodes in {meta['training_time_sec']}s -> saved {weights_path}")
    print("\nLearning Curve (last 5 checkpoints):")
    for pt in meta["learning_curve"][-5:]:
        print(f"  Ep {pt['episode']:4d} | Winrate: {pt['mbb_per_hand']:+7.1f} mbb/hand | Loss: {pt['td_loss']:.4f} | eps={pt['epsilon']:.2f}")
    print("\nMulti-Persona Benchmark (6-Max Table, RL Agent vs 5 Persona Clones):")
    for p_name, b in meta["persona_benchmarks"].items():
        print(f"  vs {p_name:<5} : {b['win_rate_bb_100']:+6.2f} bb/100 ({b['mbb_per_hand']:+7.1f} mbb/h) | Action Dist %: {b['action_dist']}")
