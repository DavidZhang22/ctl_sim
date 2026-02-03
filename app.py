# app.py
# Run:
#   pip install streamlit pandas
#   streamlit run app.py

from __future__ import annotations

import re
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple, Any, Optional, Callable

import pandas as pd
import streamlit as st

from solver import TeamMatchSolver, iter_bits, ASSUMED_RD


SHEET_DEFAULT_URL = "https://docs.google.com/spreadsheets/d/1JQePUvzoWLdC3u_CRSMEJ9bCll5qPkez0ysqL8U7YWE/edit?gid=111307365#gid=111307365"


def parse_google_sheet_url(url: str) -> Tuple[str, int]:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("Could not find spreadsheet ID in URL.")
    sheet_id = m.group(1)

    gid: Optional[int] = None
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "gid" in qs and qs["gid"]:
        gid = int(qs["gid"][0])

    if gid is None and parsed.fragment:
        frag_qs = parse_qs(parsed.fragment)
        if "gid" in frag_qs and frag_qs["gid"]:
            gid = int(frag_qs["gid"][0])

    if gid is None:
        gid = 0
    return sheet_id, gid


@st.cache_data(ttl=300)
def load_sheet_csv(sheet_id: str, gid: int) -> pd.DataFrame:
    # Works for viewable sheets (public / link-access).
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def coerce_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def extract_team_from_row(row: pd.Series, include_subs: bool) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Returns (university_name, [(player_name, glicko), ...]).
    If include_subs=True, returns top 5 by glicko among players+subs.
    """
    uni = str(row.get("University", "")).strip()
    entries: List[Tuple[str, float]] = []

    def add(name_key: str, glicko_key: str):
        name = row.get(name_key, "")
        g = row.get(glicko_key, None)
        if pd.isna(name) or str(name).strip() == "":
            return
        gf = coerce_float(g)
        if gf is None:
            return
        entries.append((str(name).strip(), gf))

    for i in range(1, 6):
        add(f"Player {i}", f"Player {i} Glicko")

    if include_subs:
        for i in range(1, 6):
            add(f"Sub {i}", f"Sub {i} Glicko")

    if include_subs:
        entries.sort(key=lambda t: t[1], reverse=True)
        entries = entries[:5]

    return uni, entries


def make_team_df(players: List[Tuple[str, float]], default_prefix: str) -> pd.DataFrame:
    # Always output 5 rows (fill placeholders if fewer)
    rows = []
    for i in range(5):
        if i < len(players):
            rows.append({"Name": players[i][0], "Glicko": float(players[i][1])})
        else:
            rows.append({"Name": f"{default_prefix}{i+1}", "Glicko": 1500.0})
    return pd.DataFrame(rows)


def validate_df(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    if "Name" not in df.columns or "Glicko" not in df.columns:
        raise ValueError("Team table must have columns Name and Glicko.")
    names = [str(x) for x in df["Name"].tolist()]
    gl = [float(x) for x in df["Glicko"].tolist()]
    if len(names) != 5 or len(gl) != 5:
        raise ValueError("Need exactly 5 rows per team.")
    return names, gl


def argmax_prob(probs: List[float]) -> int:
    return max(range(len(probs)), key=lambda i: probs[i])


def dist_table(actions: List[Any], probs: List[float], label_fn: Callable[[Any], str]) -> pd.DataFrame:
    rows = [{"Action": label_fn(a), "Prob": probs[i]} for i, a in enumerate(actions)]
    return pd.DataFrame(rows).sort_values("Prob", ascending=False).reset_index(drop=True)


def ban_pair_label(pair: Tuple[int, int], opp_names: List[str]) -> str:
    return f"{opp_names[pair[0]]} & {opp_names[pair[1]]}"


def best_seq_picks(solver: TeamMatchSolver, round_idx: int, maskA: int, maskB: int, first_picker: int) -> Tuple[int, int]:
    """Deterministic minimax picks for sequential rounds (2-5)."""
    eligA = maskA
    eligB = maskB

    if first_picker == 0:
        best_val = -1e100
        best_a = best_b = None
        for a in iter_bits(eligA):
            worst_val = 1e100
            worst_b = None
            for b in iter_bits(eligB):
                val = solver.pair_value(round_idx, maskA, maskB, a, b)
                if val < worst_val:
                    worst_val = val
                    worst_b = b
            if worst_val > best_val:
                best_val = worst_val
                best_a, best_b = a, worst_b
        return int(best_a), int(best_b)

    best_for_B = 1e100
    best_b = best_a = None
    for b in iter_bits(eligB):
        best_resp_A = -1e100
        best_a_for_b = None
        for a in iter_bits(eligA):
            val = solver.pair_value(round_idx, maskA, maskB, a, b)
            if val > best_resp_A:
                best_resp_A = val
                best_a_for_b = a
        if best_resp_A < best_for_B:
            best_for_B = best_resp_A
            best_b, best_a = b, best_a_for_b
    return int(best_a), int(best_b)


# =========================
# UI
# =========================

st.set_page_config(page_title="5v5 Draft Optimizer (FT7)", layout="wide")
st.title("5v5 match optimizer (1v1 FT7) — select teams by Google Sheet row")

with st.sidebar:
    st.header("Model settings")
    st.write(f"Winrate model: **Glicko-1 expected score** with **RD = {ASSUMED_RD:.0f}** (assumed for all players).")

    objective = st.selectbox(
        "Objective (Team A perspective)",
        ["diff", "points"],
        index=0,
    )

    interpret = st.selectbox(
        "Interpret Glicko winrate as",
        ["match", "point"],
        index=0,
        help="match (recommended): treat Glicko as FT7 match win prob, then convert to per-point prob for expected score. "
             "point: treat Glicko as per-point win prob directly."
    )

st.subheader("Team input source")

mode = st.radio("Choose team input mode", ["From Google Sheet row", "Manual"], horizontal=True)
include_subs = st.checkbox("Include substitutes (use top 5 Glicko among starters + subs)", value=False)

teamA_df: pd.DataFrame
teamB_df: pd.DataFrame
teamA_title = "Team A"
teamB_title = "Team B"

if mode == "From Google Sheet row":
    with st.expander("Google Sheet settings", expanded=True):
        sheet_url = st.text_input("Google Sheet URL", value=SHEET_DEFAULT_URL)
        try:
            sheet_id, default_gid = parse_google_sheet_url(sheet_url)
        except Exception:
            sheet_id, default_gid = ("", 0)

        gid = st.number_input("gid (tab id)", min_value=0, value=int(default_gid), step=1)

        c1, c2 = st.columns(2)
        with c1:
            rowA = st.number_input("Row for Team A (sheet row; header is row 1)", min_value=2, value=2, step=1)
        with c2:
            rowB = st.number_input("Row for Team B (sheet row; header is row 1)", min_value=2, value=3, step=1)

        load_btn = st.button("Load teams from sheet")

    if "sheet_df" not in st.session_state:
        st.session_state["sheet_df"] = None

    if load_btn:
        try:
            if not sheet_id:
                raise ValueError("Invalid sheet URL.")
            st.session_state["sheet_df"] = load_sheet_csv(sheet_id, int(gid))
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")
            st.session_state["sheet_df"] = None

    df_sheet = st.session_state.get("sheet_df", None)
    if df_sheet is None:
        st.info("Click **Load teams from sheet** (sheet/tab must be viewable).")
        st.stop()

    idxA = int(rowA) - 2  # row 2 -> df index 0
    idxB = int(rowB) - 2
    if idxA < 0 or idxA >= len(df_sheet) or idxB < 0 or idxB >= len(df_sheet):
        st.error("Row number out of range for the loaded tab.")
        st.stop()

    rowA_data = df_sheet.iloc[idxA]
    rowB_data = df_sheet.iloc[idxB]

    teamA_uni, teamA_players = extract_team_from_row(rowA_data, include_subs=include_subs)
    teamB_uni, teamB_players = extract_team_from_row(rowB_data, include_subs=include_subs)

    teamA_title = teamA_uni or "Team A"
    teamB_title = teamB_uni or "Team B"

    if len(teamA_players) < 5 or len(teamB_players) < 5:
        st.warning("One of the selected rows has < 5 valid players; missing slots are filled with placeholders.")

    teamA_df = make_team_df(teamA_players, "A")
    teamB_df = make_team_df(teamB_players, "B")

    st.subheader("Selected teams (auto-filled)")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"**{teamA_title}** (row {rowA})")
        st.dataframe(teamA_df, use_container_width=True, hide_index=True)
    with cc2:
        st.markdown(f"**{teamB_title}** (row {rowB})")
        st.dataframe(teamB_df, use_container_width=True, hide_index=True)

    if st.checkbox("Allow manual override (edit loaded teams)", value=False):
        cc3, cc4 = st.columns(2)
        with cc3:
            teamA_df = st.data_editor(teamA_df, num_rows="fixed", key="teamA_edit")
        with cc4:
            teamB_df = st.data_editor(teamB_df, num_rows="fixed", key="teamB_edit")

else:
    st.subheader("Manual team entry")
    defaultA = pd.DataFrame({"Name": [f"A{i+1}" for i in range(5)], "Glicko": [1500]*5})
    defaultB = pd.DataFrame({"Name": [f"B{i+1}" for i in range(5)], "Glicko": [1500]*5})
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Team A**")
        teamA_df = st.data_editor(defaultA, num_rows="fixed", key="teamA_manual")
    with cc2:
        st.markdown("**Team B**")
        teamB_df = st.data_editor(defaultB, num_rows="fixed", key="teamB_manual")


st.divider()
run = st.button("Compute optimal strategies", type="primary")

if run:
    try:
        namesA, ratingsA = validate_df(teamA_df)
        namesB, ratingsB = validate_df(teamB_df)
    except Exception as e:
        st.error(f"Input error: {e}")
        st.stop()

    solver = TeamMatchSolver(
        ratingsA=ratingsA,
        ratingsB=ratingsB,
        objective=objective,
        interpret_glicko_as=interpret,
    )

    # Round 1 equilibrium
    v_start, p_banA, p_banB, A_ban_pairs, B_ban_pairs, pick_strats = solver.solve_round1()

    st.subheader("Round 1 equilibrium (two bans each, then blind pick)")
    cA, cB = st.columns(2)
    with cA:
        st.markdown(f"**{teamA_title} bans (2 of {teamB_title}) — Round 1 only**")
        st.dataframe(dist_table(A_ban_pairs, p_banA, lambda pair: ban_pair_label(pair, namesB)), use_container_width=True, hide_index=True)
    with cB:
        st.markdown(f"**{teamB_title} bans (2 of {teamA_title}) — Round 1 only**")
        st.dataframe(dist_table(B_ban_pairs, p_banB, lambda pair: ban_pair_label(pair, namesA)), use_container_width=True, hide_index=True)

    st.markdown(f"**Game value from start (Team A objective = `{objective}`):** {v_start:.4f}")

    # Deterministic playthrough based on highest-prob equilibrium actions
    banB_pair = A_ban_pairs[argmax_prob(p_banA)]
    banA_pair = B_ban_pairs[argmax_prob(p_banB)]
    v_pick, p_pickA, p_pickB, A_picks, B_picks = pick_strats[(banB_pair, banA_pair)]

    st.subheader("Round 1 blind-pick equilibrium (conditioned on most-likely ban pair)")
    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown(f"**{teamA_title} blind-pick mix** (bans in effect: {ban_pair_label(banB_pair, namesB)} / {ban_pair_label(banA_pair, namesA)})")
        st.dataframe(dist_table(A_picks, p_pickA, lambda i: namesA[i]), use_container_width=True, hide_index=True)
    with cp2:
        st.markdown(f"**{teamB_title} blind-pick mix**")
        st.dataframe(dist_table(B_picks, p_pickB, lambda i: namesB[i]), use_container_width=True, hide_index=True)

    pickA_r1 = A_picks[argmax_prob(p_pickA)]
    pickB_r1 = B_picks[argmax_prob(p_pickB)]

    st.subheader("Concrete optimal playthrough (deterministic from equilibrium)")
    fullA = (1 << 5) - 1
    fullB = (1 << 5) - 1
    maskA = fullA
    maskB = fullB

    play_rows = []
    round_details = []

    # Round 1
    stats1 = solver.pair_display_stats(1, maskA, maskB, pickA_r1, pickB_r1)
    play_rows.append({
        "Round": 1,
        "Format": "2 bans + blind pick",
        "A bans (R1)": ban_pair_label(banB_pair, namesB),
        "B bans (R1)": ban_pair_label(banA_pair, namesA),
        "Pick order": "simultaneous",
        "Team A pick": namesA[pickA_r1],
        "Team B pick": namesB[pickB_r1],
        "Original E[A pts]": stats1["Ea_orig"],
        "Original E[B pts]": stats1["Eb_orig"],
        "Recorded E[A pts]": stats1["Ea_rec"],
        "Recorded E[B pts]": stats1["Eb_rec"],
        "P(buy occurs)": stats1["p_buy"],
        "P(recorded winner=A)": stats1["p_recorded_A_winner"],
        "Most likely original": f"{stats1['mode_orig'][0]}-{stats1['mode_orig'][1]} ({stats1['mode_orig'][2]:.3f})",
    })
    round_details.append(("Round 1 details (buy decisions per final score)", pd.DataFrame(stats1["details_rows"])))

    maskA &= ~(1 << pickA_r1)
    maskB &= ~(1 << pickB_r1)
    prev_winner = 0 if stats1["p_recorded_A_winner"] >= 0.5 else 1

    # Rounds 2-5: no bans, sequential pick only
    for rnd in [2, 3, 4, 5]:
        a_pick, b_pick = best_seq_picks(solver, rnd, maskA, maskB, first_picker=prev_winner)
        stats = solver.pair_display_stats(rnd, maskA, maskB, a_pick, b_pick)

        play_rows.append({
            "Round": rnd,
            "Format": "sequential pick (no bans)",
            "A bans (R1)": "",
            "B bans (R1)": "",
            "Pick order": "A first" if prev_winner == 0 else "B first",
            "Team A pick": namesA[a_pick],
            "Team B pick": namesB[b_pick],
            "Original E[A pts]": stats["Ea_orig"],
            "Original E[B pts]": stats["Eb_orig"],
            "Recorded E[A pts]": stats["Ea_rec"],
            "Recorded E[B pts]": stats["Eb_rec"],
            "P(buy occurs)": stats["p_buy"],
            "P(recorded winner=A)": stats["p_recorded_A_winner"],
            "Most likely original": f"{stats['mode_orig'][0]}-{stats['mode_orig'][1]} ({stats['mode_orig'][2]:.3f})",
        })
        round_details.append((f"Round {rnd} details (buy decisions per final score)", pd.DataFrame(stats["details_rows"])))

        maskA &= ~(1 << a_pick)
        maskB &= ~(1 << b_pick)
        prev_winner = 0 if stats["p_recorded_A_winner"] >= 0.5 else 1

    df_play = pd.DataFrame(play_rows)
    st.dataframe(df_play, use_container_width=True, hide_index=True)

    total_orig_A = float(df_play["Original E[A pts]"].sum())
    total_orig_B = float(df_play["Original E[B pts]"].sum())
    total_rec_A = float(df_play["Recorded E[A pts]"].sum())
    total_rec_B = float(df_play["Recorded E[B pts]"].sum())

    st.subheader("Final expected totals (this playthrough)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original totals (before buy rewrites):**")
        st.write(f"{teamA_title}: {total_orig_A:.2f} / 35")
        st.write(f"{teamB_title}: {total_orig_B:.2f} / 35")
    with col2:
        st.markdown("**Recorded totals (after optimal buy rewrites):**")
        st.write(f"{teamA_title}: {total_rec_A:.2f} / 35")
        st.write(f"{teamB_title}: {total_rec_B:.2f} / 35")

    st.markdown(f"**Recorded differential (A - B):** {total_rec_A - total_rec_B:.2f}")

    st.subheader("Per-round score outcomes and buy decisions")
    for title, ddf in round_details:
        with st.expander(title, expanded=False):
            st.dataframe(ddf.sort_values("Prob", ascending=False), use_container_width=True, hide_index=True)
