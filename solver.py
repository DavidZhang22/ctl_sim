# solver.py
from __future__ import annotations

import math
import itertools
from functools import lru_cache
from typing import List, Tuple, Dict, Any


# =========================
# Config
# =========================

ASSUMED_RD: float = 80.0  # everyone has RD=80 (Glicko-1)


# =========================
# Bit helpers
# =========================

def iter_bits(mask: int):
    """Yield indices of 1-bits in mask."""
    while mask:
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)
        yield i
        mask ^= lsb


# =========================
# Glicko-1 win probability (expected score)
# =========================
# E = 1 / (1 + 10^(-g(RD_j) * (R_i - R_j)/400))
# g(RD) = 1 / sqrt(1 + 3*q^2*RD^2/pi^2), q = ln(10)/400
#
# With no draws, we treat E as P(i wins).

def glicko1_g(rd: float) -> float:
    q = math.log(10.0) / 400.0
    return 1.0 / math.sqrt(1.0 + (3.0 * (q * q) * (rd * rd)) / (math.pi * math.pi))


def glicko1_win_prob(ri: float, rj: float, rd_j: float = ASSUMED_RD) -> float:
    g = glicko1_g(rd_j)
    expo = -g * (ri - rj) / 400.0
    p = 1.0 / (1.0 + (10.0 ** expo))
    eps = 1e-12
    return max(eps, min(1.0 - eps, p))


# =========================
# FT7 distribution (closed form) given per-point probability q
# =========================
# If q = P(A wins a point):
# P(A wins match 7-k) = C(6+k, k) * q^7 * (1-q)^k, k=0..6
# P(B wins match k-7) = C(6+k, k) * q^k * (1-q)^7, k=0..6

def pA_wins_ft7_from_q(q: float) -> float:
    s = 0.0
    for k in range(7):
        s += math.comb(6 + k, k) * (q ** 7) * ((1.0 - q) ** k)
    return s


def invert_match_prob_to_point_prob(p_target: float) -> float:
    """
    Find per-point probability q such that P(A wins FT7) ~= p_target.
    This maps a match-level win probability into a point-level probability
    so we can compute expected FT7 scores.
    """
    lo, hi = 1e-6, 1.0 - 1e-6
    for _ in range(80):
        mid = (lo + hi) / 2.0
        p_mid = pA_wins_ft7_from_q(mid)
        if p_mid < p_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ft7_score_distribution(q: float) -> List[Tuple[int, int, float]]:
    """List of terminal scores (a_pts, b_pts, prob)."""
    out: List[Tuple[int, int, float]] = []
    for k in range(7):
        pA = math.comb(6 + k, k) * (q ** 7) * ((1.0 - q) ** k)
        out.append((7, k, pA))
        pB = math.comb(6 + k, k) * (q ** k) * ((1.0 - q) ** 7)
        out.append((k, 7, pB))

    s = sum(p for _, _, p in out)
    if s > 0:
        out = [(a, b, p / s) for (a, b, p) in out]
    return out


def score_expectations_from_dist(dist: List[Tuple[int, int, float]]) -> Tuple[float, float, float]:
    Ea = sum(a * p for a, b, p in dist)
    Eb = sum(b * p for a, b, p in dist)
    pAwin = sum(p for a, b, p in dist if a == 7)
    return pAwin, Ea, Eb


def outcome_mode(dist: List[Tuple[int, int, float]]) -> Tuple[int, int, float]:
    a, b, p = max(dist, key=lambda x: x[2])
    return a, b, p


# =========================
# Zero-sum game solver
# =========================
# Exact support enumeration for <=6x6, regret matching for larger (e.g. 10x10 ban stage).

def _gauss_solve(M: List[List[float]], b: List[float], tol: float = 1e-12) -> List[float]:
    n = len(M)
    A = [row[:] for row in M]
    bb = b[:]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < tol:
            raise ValueError("Singular matrix")

        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            bb[col], bb[pivot] = bb[pivot], bb[col]

        piv = A[col][col]
        invp = 1.0 / piv
        for j in range(col, n):
            A[col][j] *= invp
        bb[col] *= invp

        for r in range(col + 1, n):
            factor = A[r][col]
            if abs(factor) < tol:
                continue
            for j in range(col, n):
                A[r][j] -= factor * A[col][j]
            bb[r] -= factor * bb[col]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = bb[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s
    return x


def _solve_zero_sum_exact(A: List[List[float]], tol: float = 1e-9) -> Tuple[float, List[float], List[float]]:
    m = len(A)
    n = len(A[0])
    kmax = min(m, n)

    def solve_row(pay: List[List[float]]) -> Tuple[float, List[float]]:
        mm = len(pay)
        nn = len(pay[0])

        best_v = -1e100
        best_p = [0.0] * mm

        for i in range(mm):
            v_i = min(pay[i][j] for j in range(nn))
            if v_i > best_v:
                best_v = v_i
                best_p = [0.0] * mm
                best_p[i] = 1.0

        rows = range(mm)
        cols = range(nn)

        for k in range(1, kmax + 1):
            for S in itertools.combinations(rows, k):
                S = list(S)
                for T in itertools.combinations(cols, k):
                    T = list(T)

                    M = [[0.0] * (k + 1) for _ in range(k + 1)]
                    b = [0.0] * (k + 1)

                    for eq, j in enumerate(T):
                        for idx, i in enumerate(S):
                            M[eq][idx] = pay[i][j]
                        M[eq][k] = -1.0
                        b[eq] = 0.0

                    for idx in range(k):
                        M[k][idx] = 1.0
                    b[k] = 1.0

                    try:
                        x = _gauss_solve(M, b)
                    except ValueError:
                        continue

                    pS = x[:k]
                    if any(pi < -tol for pi in pS):
                        continue

                    pS = [max(0.0, pi) for pi in pS]
                    s = sum(pS)
                    if s <= tol:
                        continue
                    pS = [pi / s for pi in pS]

                    u = []
                    for j in range(nn):
                        uj = 0.0
                        for idx, i in enumerate(S):
                            uj += pS[idx] * pay[i][j]
                        u.append(uj)

                    v = min(u)
                    if any(uj < v - 1e-6 for uj in u):
                        continue

                    if v > best_v + 1e-9:
                        best_v = v
                        best_p = [0.0] * mm
                        for idx, i in enumerate(S):
                            best_p[i] = pS[idx]

        return best_v, best_p

    v, p = solve_row(A)
    AT = [[-A[i][j] for i in range(m)] for j in range(n)]
    w, q = solve_row(AT)
    return v, p, q


def _regret_matching(A: List[List[float]], iters: int = 40000) -> Tuple[float, List[float], List[float]]:
    m = len(A)
    n = len(A[0])

    reg_r = [0.0] * m
    reg_c = [0.0] * n
    sum_r = [0.0] * m
    sum_c = [0.0] * n

    def dist_from_reg(ret: List[float]) -> List[float]:
        pos = [max(0.0, x) for x in ret]
        s = sum(pos)
        if s <= 1e-12:
            return [1.0 / len(ret)] * len(ret)
        return [x / s for x in pos]

    for _ in range(iters):
        p = dist_from_reg(reg_r)
        q = dist_from_reg(reg_c)

        for i in range(m):
            sum_r[i] += p[i]
        for j in range(n):
            sum_c[j] += q[j]

        u_row = [0.0] * m
        for i in range(m):
            s = 0.0
            Ai = A[i]
            for j in range(n):
                s += Ai[j] * q[j]
            u_row[i] = s

        u = 0.0
        for i in range(m):
            u += p[i] * u_row[i]

        v_col = [0.0] * n
        for j in range(n):
            s = 0.0
            for i in range(m):
                s += p[i] * A[i][j]
            v_col[j] = s

        for i in range(m):
            reg_r[i] += u_row[i] - u

        for j in range(n):
            reg_c[j] += u - v_col[j]

    p_avg = [x / iters for x in sum_r]
    q_avg = [x / iters for x in sum_c]

    v = 0.0
    for i in range(m):
        for j in range(n):
            v += p_avg[i] * A[i][j] * q_avg[j]
    return v, p_avg, q_avg


def solve_zero_sum_game(A: List[List[float]]) -> Tuple[float, List[float], List[float]]:
    m = len(A)
    n = len(A[0]) if m else 0
    if m == 0 or n == 0:
        raise ValueError("Empty payoff matrix")
    if any(len(row) != n for row in A):
        raise ValueError("Ragged payoff matrix")

    if max(m, n) <= 6:
        return _solve_zero_sum_exact(A)
    return _regret_matching(A, iters=40000)


# =========================
# Main game solver
# =========================

PickStrat = Tuple[float, List[float], List[float], List[int], List[int]]


class TeamMatchSolver:
    """
    Rules:
    - Round 1: each team bans 2 opponents (Round 1 only), then blind pick.
    - Rounds 2-5: no bans, sequential pick; previous *recorded* winner picks first.
    - Buy counterpick: after a match (Rounds 1-4), actual winner may rewrite that match
      to 6-7 (buyer gets 6), flipping recorded winner for next round pick order.
    - Winrate between players: Glicko-1 expected score with RD=80 for everyone.
    - Expected FT7 score: convert match winrate -> per-point winrate by inversion (default).
    """

    def __init__(
        self,
        ratingsA: List[float],
        ratingsB: List[float],
        objective: str = "diff",
        interpret_glicko_as: str = "match",  # "match" (default) or "point"
    ):
        if len(ratingsA) != 5 or len(ratingsB) != 5:
            raise ValueError("Need exactly 5 ratings per team.")
        if objective not in ("diff", "points"):
            raise ValueError("objective must be 'diff' or 'points'")
        if interpret_glicko_as not in ("match", "point"):
            raise ValueError("interpret_glicko_as must be 'match' or 'point'")

        self.rA = list(map(float, ratingsA))
        self.rB = list(map(float, ratingsB))
        self.objective = objective
        self.interpret_glicko_as = interpret_glicko_as

        self.dist: List[List[List[Tuple[int, int, float]]]] = [[[] for _ in range(5)] for __ in range(5)]
        for i in range(5):
            for j in range(5):
                p_match = glicko1_win_prob(self.rA[i], self.rB[j], rd_j=ASSUMED_RD)
                q_point = invert_match_prob_to_point_prob(p_match) if interpret_glicko_as == "match" else p_match
                self.dist[i][j] = ft7_score_distribution(q_point)

    def immediate_value(self, a_pts: int, b_pts: int) -> float:
        return float(a_pts - b_pts) if self.objective == "diff" else float(a_pts)

    @lru_cache(None)
    def solve_state(self, round_idx: int, maskA: int, maskB: int, prev_winner: int) -> float:
        if round_idx == 6:
            return 0.0
        if round_idx in (2, 3, 4, 5):
            return self.solve_seq(round_idx, maskA, maskB, maskA, maskB, first_picker=prev_winner)
        raise ValueError("Round 1 must be solved via solve_round1().")

    def buy_decision_for_outcome(
        self,
        round_idx: int,
        newMaskA: int,
        newMaskB: int,
        actual_a: int,
        actual_b: int,
    ) -> Tuple[bool, float]:
        """
        Buy allowed only after rounds 1-4. Round 5 cannot buy.
        """
        can_buy = (round_idx < 5)
        is_last = (round_idx == 5)

        if not can_buy:
            keep_immediate = self.immediate_value(actual_a, actual_b)
            keep_cont = 0.0 if is_last else self.solve_state(
                round_idx + 1, newMaskA, newMaskB, prev_winner=(0 if actual_a == 7 else 1)
            )
            return False, keep_immediate + keep_cont

        if actual_a == 7:
            # A actually won: A chooses max
            keep_immediate = self.immediate_value(actual_a, actual_b)
            keep_cont = 0.0 if is_last else self.solve_state(round_idx + 1, newMaskA, newMaskB, prev_winner=0)
            keep_total = keep_immediate + keep_cont

            buy_immediate = self.immediate_value(6, 7)  # buyer(A) gets 6
            buy_cont = 0.0 if is_last else self.solve_state(round_idx + 1, newMaskA, newMaskB, prev_winner=1)
            buy_total = buy_immediate + buy_cont

            if buy_total > keep_total + 1e-12:
                return True, buy_total
            return False, keep_total

        # B actually won: B chooses min (to minimize A)
        keep_immediate = self.immediate_value(actual_a, actual_b)
        keep_cont = 0.0 if is_last else self.solve_state(round_idx + 1, newMaskA, newMaskB, prev_winner=1)
        keep_total = keep_immediate + keep_cont

        buy_immediate = self.immediate_value(7, 6)  # buyer(B) gets 6 => recorded winner=A
        buy_cont = 0.0 if is_last else self.solve_state(round_idx + 1, newMaskA, newMaskB, prev_winner=0)
        buy_total = buy_immediate + buy_cont

        if buy_total < keep_total - 1e-12:
            return True, buy_total
        return False, keep_total

    @lru_cache(None)
    def pair_value(self, round_idx: int, maskA: int, maskB: int, a: int, b: int) -> float:
        newA = maskA & ~(1 << a)
        newB = maskB & ~(1 << b)

        total = 0.0
        for aa, bb, p in self.dist[a][b]:
            _, chosen = self.buy_decision_for_outcome(round_idx, newA, newB, aa, bb)
            total += p * chosen
        return total

    def pair_display_stats(self, round_idx: int, maskA: int, maskB: int, a: int, b: int) -> Dict[str, Any]:
        newA = maskA & ~(1 << a)
        newB = maskB & ~(1 << b)

        dist = self.dist[a][b]
        pAwin, Ea_orig, Eb_orig = score_expectations_from_dist(dist)
        mode_a, mode_b, mode_p = outcome_mode(dist)

        Ea_rec = Eb_rec = p_buy = p_recorded_A_winner = 0.0
        details_rows: List[Dict[str, Any]] = []

        for aa, bb, p in dist:
            buy, _ = self.buy_decision_for_outcome(round_idx, newA, newB, aa, bb)
            if buy:
                p_buy += p
                ra, rb = (6, 7) if aa == 7 else (7, 6)
            else:
                ra, rb = aa, bb

            Ea_rec += ra * p
            Eb_rec += rb * p
            if ra > rb:
                p_recorded_A_winner += p

            details_rows.append({
                "Original score": f"{aa}-{bb}",
                "Prob": p,
                "Actual winner": "A" if aa == 7 else "B",
                "Buy counterpick?": "YES" if buy else "no",
                "Recorded score": f"{ra}-{rb}",
                "Recorded winner": "A" if ra > rb else "B",
            })

        return {
            "pAwin_actual": pAwin,
            "Ea_orig": Ea_orig,
            "Eb_orig": Eb_orig,
            "Ea_rec": Ea_rec,
            "Eb_rec": Eb_rec,
            "p_buy": p_buy,
            "p_recorded_A_winner": p_recorded_A_winner,
            "mode_orig": (mode_a, mode_b, mode_p),
            "details_rows": details_rows,
        }

    def solve_seq(self, round_idx: int, maskA: int, maskB: int, eligA: int, eligB: int, first_picker: int) -> float:
        if first_picker == 0:
            best = -1e100
            for a in iter_bits(eligA):
                worst = 1e100
                for b in iter_bits(eligB):
                    worst = min(worst, self.pair_value(round_idx, maskA, maskB, a, b))
                best = max(best, worst)
            return best
        else:
            best_for_B = 1e100
            for b in iter_bits(eligB):
                best_resp_A = -1e100
                for a in iter_bits(eligA):
                    best_resp_A = max(best_resp_A, self.pair_value(round_idx, maskA, maskB, a, b))
                best_for_B = min(best_for_B, best_resp_A)
            return best_for_B

    def solve_round1(self):
        """
        Round 1:
          - simultaneous TWO bans per team (Round 1 only)
          - blind pick (simultaneous)

        Returns:
          v_ban, pA_ban, pB_ban, A_ban_pairs, B_ban_pairs, pick_strats
        """
        fullA = (1 << 5) - 1
        fullB = (1 << 5) - 1

        A_single = list(iter_bits(fullB))
        B_single = list(iter_bits(fullA))

        A_ban_pairs = list(itertools.combinations(A_single, 2))  # 10
        B_ban_pairs = list(itertools.combinations(B_single, 2))  # 10

        ban_payoff = [[0.0 for _ in B_ban_pairs] for __ in A_ban_pairs]
        pick_strats: Dict[Tuple[Tuple[int, int], Tuple[int, int]], PickStrat] = {}

        for i, banB_pair in enumerate(A_ban_pairs):
            for j, banA_pair in enumerate(B_ban_pairs):
                # bans apply ONLY to Round 1 eligibility
                eligA = fullA & ~(1 << banA_pair[0]) & ~(1 << banA_pair[1])
                eligB = fullB & ~(1 << banB_pair[0]) & ~(1 << banB_pair[1])

                A_picks = list(iter_bits(eligA))
                B_picks = list(iter_bits(eligB))

                P = [[0.0 for _ in B_picks] for __ in A_picks]
                for ia, a in enumerate(A_picks):
                    for ib, b in enumerate(B_picks):
                        P[ia][ib] = self.pair_value(1, fullA, fullB, a, b)

                v_pick, p_pickA, p_pickB = solve_zero_sum_game(P)
                ban_payoff[i][j] = v_pick
                pick_strats[(banB_pair, banA_pair)] = (v_pick, p_pickA, p_pickB, A_picks, B_picks)

        v_ban, pA_ban, pB_ban = solve_zero_sum_game(ban_payoff)
        return v_ban, pA_ban, pB_ban, A_ban_pairs, B_ban_pairs, pick_strats
