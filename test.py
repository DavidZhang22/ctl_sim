# app.py
# Streamlit UI for 5v5 match of FT7 1v1s with:
# - Round 1: simultaneous ban + blind pick
# - Round 2: simultaneous ban + sequential pick (previous recorded winner picks first)
# - Rounds 3-5: sequential pick (previous recorded winner picks first)
# - After each match (Rounds 1-4), the ACTUAL winner may "buy counterpick" by rewriting
#   THAT match's recorded score to 6-7 (buyer gets 6), which flips who is the recorded winner
#   and thus flips who picks first next round.
#
# Additional rule:
# - The Round 2 ban cannot be the same player that team banned in Round 1.
#
# Run:
#   pip install streamlit pandas
#   streamlit run app.py

from __future__ import annotations

import math
import itertools
from functools import lru_cache
from typing import List, Tuple, Dict

import streamlit as st
import pandas as pd


# =========================
# 1) Rating -> win chance
# =========================

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def glicko_to_win_prob(ra: float, rb: float, sigma: float) -> float:
    """
    sigma rating difference ~ 1 std dev in win chance.
    Probit model:
      P(A wins) = Φ((ra-rb)/sigma)
    """
    p = normal_cdf((ra - rb) / sigma)
    eps = 1e-12
    return max(eps, min(1.0 - eps, p))


# =========================
# 2) FT7 distribution (closed form)
# =========================
# If q = P(A wins an individual point), then:
# P(A wins match 7-k) = C(6+k, k) * q^7 * (1-q)^k, for k=0..6
# P(B wins match k-7) = C(6+k, k) * q^k * (1-q)^7, for k=0..6


def pA_wins_ft7_from_q(q: float) -> float:
    s = 0.0
    for k in range(7):
        s += math.comb(6 + k, k) * (q ** 7) * ((1.0 - q) ** k)
    return s


def invert_match_prob_to_point_prob(p_target: float) -> float:
    """
    Given a target match win probability p_target for FT7, find point win probability q
    such that P(A wins FT7) ~= p_target.
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
    """
    Return list of (a_points, b_points, prob) for terminal FT7 scores.
    a_points=7,b<=6 or b_points=7,a<=6.
    """
    out = []
    for k in range(7):
        pA = math.comb(6 + k, k) * (q ** 7) * ((1.0 - q) ** k)
        out.append((7, k, pA))
        pB = math.comb(6 + k, k) * (q ** k) * ((1.0 - q) ** 7)
        out.append((k, 7, pB))
    # Normalize tiny drift
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
# 3) Small zero-sum solver (support enumeration)
# =========================

def gauss_solve(M: List[List[float]], b: List[float], tol: float = 1e-12) -> List[float]:
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


def solve_zero_sum_game(A: List[List[float]], tol: float = 1e-9) -> Tuple[float, List[float], List[float]]:
    """
    Solve a small zero-sum game for row player (maximizer).
    Returns:
      value v, row_mix p, col_mix q
    """
    m = len(A)
    n = len(A[0]) if m else 0
    if m == 0 or n == 0:
        raise ValueError("Empty payoff matrix")
    if any(len(row) != n for row in A):
        raise ValueError("Ragged payoff matrix")

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

                    # unknowns: p_S (k) and v (1) => k+1
                    M = [[0.0] * (k + 1) for _ in range(k + 1)]
                    b = [0.0] * (k + 1)

                    # For each j in T: sum_i p_i * pay[i][j] = v
                    for eq, j in enumerate(T):
                        for idx, i in enumerate(S):
                            M[eq][idx] = pay[i][j]
                        M[eq][k] = -1.0
                        b[eq] = 0.0

                    # sum p_i = 1
                    for idx in range(k):
                        M[k][idx] = 1.0
                    b[k] = 1.0

                    try:
                        x = gauss_solve(M, b)
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

                    # Check min payoff
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

    # Column player minimizes A, equivalent to row player maximizing (-A^T)
    AT = [[-A[i][j] for i in range(m)] for j in range(n)]
    w, q = solve_row(AT)
    return v, p, q


# =========================
# 4) Game solver with "buy rewrites last match to 6-7"
# =========================

def _bits(mask: int):
    while mask:
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)
        yield i
        mask ^= lsb


class TeamMatchSolver:
    """
    Solves optimal play from Team A perspective.

    buy rule:
      After a match is played, the actual winner may buy counterpick by rewriting THAT match's
      recorded score to 6-7, with the buyer getting 6 (so recorded winner flips).
      The buy decision can depend on the realized final score (7-k vs 7-6 etc.).

    additional ban rule:
      Round 2 ban cannot be the same player that team banned in Round 1.
    """

    def __init__(
        self,
        ratingsA: List[float],
        ratingsB: List[float],
        sigma: float,
        objective: str = "diff",
        rating_model: str = "match",
    ):
        if len(ratingsA) != 5 or len(ratingsB) != 5:
            raise ValueError("Need exactly 5 ratings per team.")
        if objective not in ("diff", "points"):
            raise ValueError("objective must be 'diff' or 'points'")
        if rating_model not in ("match", "point"):
            raise ValueError("rating_model must be 'match' or 'point'")

        self.rA = list(map(float, ratingsA))
        self.rB = list(map(float, ratingsB))
        self.sigma = float(sigma)
        self.objective = objective
        self.rating_model = rating_model

        # Precompute per-point q and FT7 distributions for each pairing
        self.q = [[0.0] * 5 for _ in range(5)]
        self.dist = [[None] * 5 for _ in range(5)]  # type: ignore
        for i in range(5):
            for j in range(5):
                p = glicko_to_win_prob(self.rA[i], self.rB[j], self.sigma)
                if self.rating_model == "point":
                    q = p
                else:
                    q = invert_match_prob_to_point_prob(p)
                self.q[i][j] = q
                self.dist[i][j] = ft7_score_distribution(q)

    def immediate_value(self, a_pts: int, b_pts: int) -> float:
        if self.objective == "diff":
            return float(a_pts - b_pts)
        return float(a_pts)

    @lru_cache(None)
    def solve_state(
        self,
        round_idx: int,
        maskA: int,
        maskB: int,
        prev_winner: int,
        ban1_A_on_B: int,
        ban1_B_on_A: int,
    ) -> float:
        if round_idx == 6:
            return 0.0
        if round_idx == 2:
            return self.solve_round2(maskA, maskB, prev_winner, ban1_A_on_B, ban1_B_on_A)
        if round_idx in (3, 4, 5):
            return self.solve_seq(
                round_idx, maskA, maskB, maskA, maskB,
                first_picker=prev_winner,
                ban1_A_on_B=ban1_A_on_B,
                ban1_B_on_A=ban1_B_on_A
            )
        raise ValueError("Round 1 must be solved via solve_round1().")

    def buy_decision_for_outcome(
        self,
        round_idx: int,
        newMaskA: int,
        newMaskB: int,
        actual_a: int,
        actual_b: int,
        ban1_A_on_B: int,
        ban1_B_on_A: int,
    ) -> Tuple[bool, float]:
        """
        Returns (buy?, chosen_total_value_for_A_from_this_outcome).

        If buy happens, recorded score becomes:
          - If A was actual winner: (6,7), next prev_winner=B
          - If B was actual winner: (7,6), next prev_winner=A
        """
        is_last = (round_idx == 5)

        if actual_a == 7:
            # A is actual winner: A chooses max
            keep_immediate = self.immediate_value(actual_a, actual_b)
            keep_cont = 0.0 if is_last else self.solve_state(
                round_idx + 1, newMaskA, newMaskB, 0, ban1_A_on_B, ban1_B_on_A
            )
            keep_total = keep_immediate + keep_cont

            buy_immediate = self.immediate_value(6, 7)
            buy_cont = 0.0 if is_last else self.solve_state(
                round_idx + 1, newMaskA, newMaskB, 1, ban1_A_on_B, ban1_B_on_A
            )
            buy_total = buy_immediate + buy_cont

            if buy_total > keep_total + 1e-12:
                return True, buy_total
            return False, keep_total

        # B is actual winner: B chooses min (to minimize A value)
        keep_immediate = self.immediate_value(actual_a, actual_b)
        keep_cont = 0.0 if is_last else self.solve_state(
            round_idx + 1, newMaskA, newMaskB, 1, ban1_A_on_B, ban1_B_on_A
        )
        keep_total = keep_immediate + keep_cont

        buy_immediate = self.immediate_value(7, 6)
        buy_cont = 0.0 if is_last else self.solve_state(
            round_idx + 1, newMaskA, newMaskB, 0, ban1_A_on_B, ban1_B_on_A
        )
        buy_total = buy_immediate + buy_cont

        if buy_total < keep_total - 1e-12:
            return True, buy_total
        return False, keep_total

    @lru_cache(None)
    def pair_value(
        self,
        round_idx: int,
        maskA: int,
        maskB: int,
        a: int,
        b: int,
        ban1_A_on_B: int,
        ban1_B_on_A: int,
    ) -> float:
        newA = maskA & ~(1 << a)
        newB = maskB & ~(1 << b)

        total = 0.0
        dist = self.dist[a][b]
        for aa, bb, p in dist:
            _, chosen = self.buy_decision_for_outcome(
                round_idx, newA, newB, aa, bb, ban1_A_on_B, ban1_B_on_A
            )
            total += p * chosen
        return total

    def pair_display_stats(
        self,
        round_idx: int,
        maskA: int,
        maskB: int,
        a: int,
        b: int,
        ban1_A_on_B: int,
        ban1_B_on_A: int,
    ) -> Dict:
        newA = maskA & ~(1 << a)
        newB = maskB & ~(1 << b)

        dist = self.dist[a][b]
        pAwin, Ea_orig, Eb_orig = score_expectations_from_dist(dist)
        mode_a, mode_b, mode_p = outcome_mode(dist)

        Ea_rec = 0.0
        Eb_rec = 0.0
        p_buy = 0.0
        p_recorded_A_winner = 0.0

        rows = []
        for aa, bb, p in dist:
            buy, _ = self.buy_decision_for_outcome(
                round_idx, newA, newB, aa, bb, ban1_A_on_B, ban1_B_on_A
            )

            if buy:
                p_buy += p
                if aa == 7:
                    ra, rb = 6, 7
                else:
                    ra, rb = 7, 6
            else:
                ra, rb = aa, bb

            Ea_rec += ra * p
            Eb_rec += rb * p
            if ra > rb:
                p_recorded_A_winner += p

            rows.append({
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
            "details_df": pd.DataFrame(rows).sort_values("Prob", ascending=False).reset_index(drop=True),
        }

    def solve_seq(
        self,
        round_idx: int,
        maskA: int,
        maskB: int,
        eligA: int,
        eligB: int,
        first_picker: int,
        ban1_A_on_B: int,
        ban1_B_on_A: int,
    ) -> float:
        if first_picker == 0:
            best = -1e100
            for a in _bits(eligA):
                worst = 1e100
                for b in _bits(eligB):
                    val = self.pair_value(round_idx, maskA, maskB, a, b, ban1_A_on_B, ban1_B_on_A)
                    worst = min(worst, val)
                best = max(best, worst)
            return best
        else:
            best_for_B = 1e100
            for b in _bits(eligB):
                best_resp_A = -1e100
                for a in _bits(eligA):
                    val = self.pair_value(round_idx, maskA, maskB, a, b, ban1_A_on_B, ban1_B_on_A)
                    best_resp_A = max(best_resp_A, val)
                best_for_B = min(best_for_B, best_resp_A)
            return best_for_B

    def solve_round2(
        self,
        maskA: int,
        maskB: int,
        prev_winner: int,
        ban1_A_on_B: int,
        ban1_B_on_A: int
    ) -> float:
        """
        Round 2: simultaneous ban (cannot repeat Round 1 ban), then sequential pick.
        """
        A_bans = [b for b in _bits(maskB) if b != ban1_A_on_B]
        B_bans = [a for a in _bits(maskA) if a != ban1_B_on_A]

        if not A_bans:
            A_bans = list(_bits(maskB))
        if not B_bans:
            B_bans = list(_bits(maskA))

        payoff = [[0.0 for _ in B_bans] for __ in A_bans]
        for i, banB in enumerate(A_bans):
            for j, banA in enumerate(B_bans):
                eligA = maskA & ~(1 << banA)
                eligB = maskB & ~(1 << banB)
                payoff[i][j] = self.solve_seq(
                    2, maskA, maskB, eligA, eligB,
                    first_picker=prev_winner,
                    ban1_A_on_B=ban1_A_on_B,
                    ban1_B_on_A=ban1_B_on_A
                )

        v, _, _ = solve_zero_sum_game(payoff)
        return v

    def solve_round1(self):
        """
        Round 1: simultaneous ban, then blind pick (simultaneous).
        Returns equilibrium for ban stage and, for each ban pair, equilibrium for blind pick.
        """
        fullA = (1 << 5) - 1
        fullB = (1 << 5) - 1

        A_bans = list(_bits(fullB))
        B_bans = list(_bits(fullA))

        ban_payoff = [[0.0 for _ in B_bans] for __ in A_bans]
        pick_strats: Dict[Tuple[int, int], Tuple[float, List[float], List[float], List[int], List[int]]] = {}

        for i, banB in enumerate(A_bans):
            for j, banA in enumerate(B_bans):
                eligA = fullA & ~(1 << banA)
                eligB = fullB & ~(1 << banB)

                A_picks = list(_bits(eligA))
                B_picks = list(_bits(eligB))

                P = [[0.0 for _ in B_picks] for __ in A_picks]
                for ia, a in enumerate(A_picks):
                    for ib, b in enumerate(B_picks):
                        # Pass Round-1 bans forward so Round-2 cannot repeat them.
                        P[ia][ib] = self.pair_value(1, fullA, fullB, a, b, banB, banA)

                v_pick, p_pick, q_pick = solve_zero_sum_game(P)
                ban_payoff[i][j] = v_pick
                pick_strats[(banB, banA)] = (v_pick, p_pick, q_pick, A_picks, B_picks)

        v_ban, p_ban, q_ban = solve_zero_sum_game(ban_payoff)
        return v_ban, p_ban, q_ban, A_bans, B_bans, pick_strats


# =========================
# 5) Deterministic "playthrough" helpers
# =========================

def argmax_prob(actions: List[int], probs: List[float]) -> int:
    return actions[max(range(len(actions)), key=lambda i: probs[i])]


def best_seq_picks(
    solver: TeamMatchSolver,
    round_idx: int,
    maskA: int,
    maskB: int,
    eligA: int,
    eligB: int,
    first_picker: int,
    ban1_A_on_B: int,
    ban1_B_on_A: int,
):
    if first_picker == 0:
        best_val = -1e100
        best_a = None
        best_b = None
        for a in _bits(eligA):
            worst_val = 1e100
            worst_b = None
            for b in _bits(eligB):
                val = solver.pair_value(round_idx, maskA, maskB, a, b, ban1_A_on_B, ban1_B_on_A)
                if val < worst_val:
                    worst_val = val
                    worst_b = b
            if worst_val > best_val:
                best_val = worst_val
                best_a, best_b = a, worst_b
        return best_a, best_b

    best_for_B = 1e100
    best_b = None
    best_a = None
    for b in _bits(eligB):
        best_resp_A = -1e100
        best_a_for_b = None
        for a in _bits(eligA):
            val = solver.pair_value(round_idx, maskA, maskB, a, b, ban1_A_on_B, ban1_B_on_A)
            if val > best_resp_A:
                best_resp_A = val
                best_a_for_b = a
        if best_resp_A < best_for_B:
            best_for_B = best_resp_A
            best_b, best_a = b, best_a_for_b
    return best_a, best_b


def round2_bans_and_picks_deterministic(
    solver: TeamMatchSolver,
    maskA: int,
    maskB: int,
    prev_winner: int,
    ban1_A_on_B: int,
    ban1_B_on_A: int,
):
    # Respect "cannot repeat round-1 ban"
    A_bans = [b for b in _bits(maskB) if b != ban1_A_on_B]
    B_bans = [a for a in _bits(maskA) if a != ban1_B_on_A]
    if not A_bans:
        A_bans = list(_bits(maskB))
    if not B_bans:
        B_bans = list(_bits(maskA))

    payoff = [[0.0 for _ in B_bans] for __ in A_bans]
    for i, banB in enumerate(A_bans):
        for j, banA in enumerate(B_bans):
            eligA = maskA & ~(1 << banA)
            eligB = maskB & ~(1 << banB)
            payoff[i][j] = solver.solve_seq(
                2, maskA, maskB, eligA, eligB,
                first_picker=prev_winner,
                ban1_A_on_B=ban1_A_on_B,
                ban1_B_on_A=ban1_B_on_A
            )

    v, pA, pB = solve_zero_sum_game(payoff)
    banB = argmax_prob(A_bans, pA)
    banA = argmax_prob(B_bans, pB)

    eligA = maskA & ~(1 << banA)
    eligB = maskB & ~(1 << banB)

    a_pick, b_pick = best_seq_picks(
        solver, 2, maskA, maskB, eligA, eligB, first_picker=prev_winner,
        ban1_A_on_B=ban1_A_on_B, ban1_B_on_A=ban1_B_on_A
    )

    return {
        "ban_value": v,
        "A_bans": A_bans,
        "B_bans": B_bans,
        "pA": pA,
        "pB": pB,
        "banB": banB,
        "banA": banA,
        "a": a_pick,
        "b": b_pick,
    }


# =========================
# 6) UI
# =========================

st.set_page_config(page_title="5v5 Draft Optimizer (FT7)", layout="wide")
st.title("5v5 match optimizer (1v1 FT7) with bans + counterpick-buy")

with st.sidebar:
    st.header("Model settings")
    sigma = st.number_input("Sigma (rating diff = 1 std dev)", min_value=1.0, value=200.0, step=10.0)
    objective = st.selectbox(
        "Objective (Team A perspective)",
        ["diff", "points"],
        index=0,
        help="diff = maximize expected (A total points - B total points). points = maximize expected A total points (B minimizes it)."
    )
    rating_model = st.selectbox(
        "Interpret Φ((ra-rb)/sigma) as",
        ["match", "point"],
        index=0,
        help="match = match win prob for FT7 (we invert to point prob). point = per-point win prob directly."
    )

st.subheader("Inputs")
col1, col2 = st.columns(2)

defaultA = pd.DataFrame({"Name": [f"A{i+1}" for i in range(5)], "Glicko": [1500, 1500, 1500, 1500, 1500]})
defaultB = pd.DataFrame({"Name": [f"B{i+1}" for i in range(5)], "Glicko": [1500, 1500, 1500, 1500, 1500]})

with col1:
    st.markdown("**Team A**")
    dfA = st.data_editor(defaultA, num_rows="fixed", key="teamA")
with col2:
    st.markdown("**Team B**")
    dfB = st.data_editor(defaultB, num_rows="fixed", key="teamB")


def validate_df(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    if "Name" not in df.columns or "Glicko" not in df.columns:
        raise ValueError("Table must have columns Name and Glicko.")
    names = [str(x) for x in df["Name"].tolist()]
    gl = [float(x) for x in df["Glicko"].tolist()]
    if len(names) != 5 or len(gl) != 5:
        raise ValueError("Need exactly 5 rows per team.")
    return names, gl


run = st.button("Compute optimal strategies")

if run:
    try:
        namesA, ratingsA = validate_df(dfA)
        namesB, ratingsB = validate_df(dfB)
    except Exception as e:
        st.error(f"Input error: {e}")
        st.stop()

    solver = TeamMatchSolver(
        ratingsA=ratingsA,
        ratingsB=ratingsB,
        sigma=sigma,
        objective=objective,
        rating_model=rating_model,
    )

    # ---------- Round 1 equilibrium ----------
    v_start, p_banA, p_banB, A_ban_actions, B_ban_actions, pick_strats = solver.solve_round1()

    st.subheader("Round 1 equilibrium (simultaneous ban, then blind pick)")

    def dist_table(actions: List[int], probs: List[float], labels: List[str]) -> pd.DataFrame:
        rows = [{"Action": labels[a], "Prob": probs[i]} for i, a in enumerate(actions)]
        return pd.DataFrame(rows).sort_values("Prob", ascending=False).reset_index(drop=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Team A ban mix (bans a Team B player from Round 1)**")
        st.dataframe(dist_table(A_ban_actions, p_banA, namesB), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Team B ban mix (bans a Team A player from Round 1)**")
        st.dataframe(dist_table(B_ban_actions, p_banB, namesA), use_container_width=True, hide_index=True)

    st.markdown(f"**Game value from start (Team A objective = `{objective}`):** {v_start:.4f}")

    # Deterministic ban resolution (highest-prob) for a concrete playthrough
    banB_r1 = argmax_prob(A_ban_actions, p_banA)  # A banned this B player in round 1
    banA_r1 = argmax_prob(B_ban_actions, p_banB)  # B banned this A player in round 1
    v_pick, p_pickA, p_pickB, A_picks, B_picks = pick_strats[(banB_r1, banA_r1)]

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            f"**Round 1 blind-pick mix for Team A** (given bans: A banned `{namesB[banB_r1]}`, B banned `{namesA[banA_r1]}`)"
        )
        st.dataframe(dist_table(A_picks, p_pickA, namesA), use_container_width=True, hide_index=True)
    with c4:
        st.markdown("**Round 1 blind-pick mix for Team B**")
        st.dataframe(dist_table(B_picks, p_pickB, namesB), use_container_width=True, hide_index=True)

    pickA_r1 = argmax_prob(A_picks, p_pickA)
    pickB_r1 = argmax_prob(B_picks, p_pickB)

    # ---------- Concrete playthrough ----------
    st.subheader("Concrete optimal playthrough (deterministic choices at mixed stages)")

    fullA = (1 << 5) - 1
    fullB = (1 << 5) - 1
    maskA = fullA
    maskB = fullB

    play_rows = []
    round_details = []

    # Round 1
    stats1 = solver.pair_display_stats(1, maskA, maskB, pickA_r1, pickB_r1, banB_r1, banA_r1)
    play_rows.append({
        "Round": 1,
        "Format": "ban + blind pick",
        "A ban": namesB[banB_r1],
        "B ban": namesA[banA_r1],
        "Pick order": "simultaneous",
        "Team A pick": namesA[pickA_r1],
        "Team B pick": namesB[pickB_r1],
        "Original E[A pts]": stats1["Ea_orig"],
        "Original E[B pts]": stats1["Eb_orig"],
        "Recorded E[A pts]": stats1["Ea_rec"],
        "Recorded E[B pts]": stats1["Eb_rec"],
        "P(buy occurs)": stats1["p_buy"],
        "P(recorded winner = A)": stats1["p_recorded_A_winner"],
        "Most likely original": f"{stats1['mode_orig'][0]}-{stats1['mode_orig'][1]} ({stats1['mode_orig'][2]:.3f})",
    })
    round_details.append(("Round 1 details (buy decisions per possible final score)", stats1["details_df"]))

    maskA &= ~(1 << pickA_r1)
    maskB &= ~(1 << pickB_r1)

    prev_winner = 0 if stats1["p_recorded_A_winner"] >= 0.5 else 1

    # Round 2 (respecting "cannot repeat round-1 ban")
    r2 = round2_bans_and_picks_deterministic(
        solver, maskA, maskB, prev_winner=prev_winner,
        ban1_A_on_B=banB_r1, ban1_B_on_A=banA_r1
    )
    stats2 = solver.pair_display_stats(2, maskA, maskB, r2["a"], r2["b"], banB_r1, banA_r1)
    play_rows.append({
        "Round": 2,
        "Format": "ban + sequential pick",
        "A ban": namesB[r2["banB"]],
        "B ban": namesA[r2["banA"]],
        "Pick order": "A first" if prev_winner == 0 else "B first",
        "Team A pick": namesA[r2["a"]],
        "Team B pick": namesB[r2["b"]],
        "Original E[A pts]": stats2["Ea_orig"],
        "Original E[B pts]": stats2["Eb_orig"],
        "Recorded E[A pts]": stats2["Ea_rec"],
        "Recorded E[B pts]": stats2["Eb_rec"],
        "P(buy occurs)": stats2["p_buy"],
        "P(recorded winner = A)": stats2["p_recorded_A_winner"],
        "Most likely original": f"{stats2['mode_orig'][0]}-{stats2['mode_orig'][1]} ({stats2['mode_orig'][2]:.3f})",
    })
    round_details.append(("Round 2 details (buy decisions per possible final score)", stats2["details_df"]))

    maskA &= ~(1 << r2["a"])
    maskB &= ~(1 << r2["b"])
    prev_winner = 0 if stats2["p_recorded_A_winner"] >= 0.5 else 1

    # Rounds 3-5
    for rnd in [3, 4, 5]:
        a_pick, b_pick = best_seq_picks(
            solver, rnd, maskA, maskB, maskA, maskB, first_picker=prev_winner,
            ban1_A_on_B=banB_r1, ban1_B_on_A=banA_r1
        )
        stats = solver.pair_display_stats(rnd, maskA, maskB, a_pick, b_pick, banB_r1, banA_r1)

        play_rows.append({
            "Round": rnd,
            "Format": "sequential pick",
            "A ban": "",
            "B ban": "",
            "Pick order": "A first" if prev_winner == 0 else "B first",
            "Team A pick": namesA[a_pick],
            "Team B pick": namesB[b_pick],
            "Original E[A pts]": stats["Ea_orig"],
            "Original E[B pts]": stats["Eb_orig"],
            "Recorded E[A pts]": stats["Ea_rec"],
            "Recorded E[B pts]": stats["Eb_rec"],
            "P(buy occurs)": stats["p_buy"],
            "P(recorded winner = A)": stats["p_recorded_A_winner"],
            "Most likely original": f"{stats['mode_orig'][0]}-{stats['mode_orig'][1]} ({stats['mode_orig'][2]:.3f})",
        })
        round_details.append((f"Round {rnd} details (buy decisions per possible final score)", stats["details_df"]))

        maskA &= ~(1 << a_pick)
        maskB &= ~(1 << b_pick)
        prev_winner = 0 if stats["p_recorded_A_winner"] >= 0.5 else 1

    df_play = pd.DataFrame(play_rows)
    st.dataframe(df_play, use_container_width=True, hide_index=True)

    total_orig_A = float(df_play["Original E[A pts]"].sum())
    total_orig_B = float(df_play["Original E[B pts]"].sum())
    total_rec_A = float(df_play["Recorded E[A pts]"].sum())
    total_rec_B = float(df_play["Recorded E[B pts]"].sum())

    st.subheader("Final expected totals (this concrete playthrough)")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Original totals (before any buy rewrites):**")
        st.write(f"Team A: {total_orig_A:.2f} / 35")
        st.write(f"Team B: {total_orig_B:.2f} / 35")
    with c6:
        st.markdown("**Recorded totals (after optimal buy rewrites):**")
        st.write(f"Team A: {total_rec_A:.2f} / 35")
        st.write(f"Team B: {total_rec_B:.2f} / 35")

    st.markdown(f"**Recorded differential (A - B):** {total_rec_A - total_rec_B:.2f}")

    st.subheader("Per-round score outcomes and buy decisions")
    st.caption(
        "Rows where **Buy counterpick? = YES** show the **original score** for that outcome "
        "and the **recorded score** after it gets rewritten to 6–7 (buyer gets 6)."
    )
    for title, ddf in round_details:
        with st.expander(title, expanded=False):
            st.dataframe(ddf, use_container_width=True, hide_index=True)
