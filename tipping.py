#!/usr/bin/env python3
"""
AFL tipping-comp optimal strategy engine.

Answers: given the leaderboard, the market odds, and the fact that a tie for first
goes to whoever holds the lowest cumulative margin error, what do I tip in the next
game that locks -- and what is P(finish first) under each option?

Pure standard library. No numpy, no pandas, no openpyxl.

Usage:
    python3 tipping.py --make-template     # write inputs/*.csv for you to fill
    python3 tipping.py --recommend         # solve and print the next decision
    python3 tipping.py --recommend --explain
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(HERE, "inputs")
OUTPUT_DIR = os.path.join(HERE, "output")

DEFAULT_SET = "current"


@dataclass(frozen=True)
class SetPaths:
    """Every path belonging to one named input set.

    A set pairs exactly one leaderboard with one fixture list, so the two can
    never be mixed across sets by accident.
    """
    name: str
    leaderboard: str
    fixtures: str
    output_dir: str


def resolve_set(name: str,
                input_dir: Optional[str] = None,
                output_dir: Optional[str] = None) -> "SetPaths":
    """Build the paths for set `name`. Pure: touches no filesystem."""
    if not name or os.sep in name or (os.altsep and os.altsep in name) \
            or name in (".", "..") or os.path.isabs(name):
        raise InputError(
            "invalid set name %r: use a plain directory name such as 'current' "
            "or 'scenario'" % name
        )
    ind = INPUT_DIR if input_dir is None else input_dir
    outd = OUTPUT_DIR if output_dir is None else output_dir
    return SetPaths(
        name=name,
        leaderboard=os.path.join(ind, name, "leaderboard.csv"),
        fixtures=os.path.join(ind, name, "fixtures.csv"),
        output_dir=os.path.join(outd, name),
    )


LEADERBOARD_CSV = os.path.join(INPUT_DIR, DEFAULT_SET, "leaderboard.csv")
FIXTURES_CSV = os.path.join(INPUT_DIR, DEFAULT_SET, "fixtures.csv")

# Margin model constants. See docs/.../design.md "Margin countback model".
SIGMA_MARGIN = 37.0   # SD of actual margin around the line
TAU_TIP = 10.0        # SD of tipsters' margin tips around the line (ASSUMED, not fitted)
DELTA_CLAMP = 20      # |delta| never needs to exceed the number of remaining games


# --------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tipster:
    name: str
    points: int
    margin_error: int
    is_me: bool = False
    joker_available: bool = False   # every tipster has spent theirs; see design doc


@dataclass(frozen=True)
class Game:
    game_id: str
    round: str
    lock_local: str
    home: str
    away: str
    home_odds: float
    away_odds: float
    line_fav: Optional[float]        # favourite's expected winning margin, positive
    is_margin_game: bool
    p_home_override: Optional[float] = None


class InputError(Exception):
    """Raised with a message naming the offending file, row and column."""


# --------------------------------------------------------------------------------
# Devigging (section 3 of the brief)
#
# All closed-form for 2-outcome markets. pi_i = 1/odds_i, B = pi_h + pi_a.
# --------------------------------------------------------------------------------

def devig_proportional(home_odds: float, away_odds: float) -> Tuple[float, float]:
    """p_i = pi_i / B. Distributes the overround in proportion to implied probability."""
    pi_h, pi_a = 1.0 / home_odds, 1.0 / away_odds
    b = pi_h + pi_a
    return pi_h / b, pi_a / b


def devig_odds_ratio(home_odds: float, away_odds: float) -> Tuple[float, float]:
    """Odds-ratio (log-odds / Cheung) devig.

    Sets p_i/(1-p_i) = c * pi_i/(1-pi_i). Summing to 1 over two outcomes gives
    c^2 * a * b = 1, hence p_h = sqrt(a) / (sqrt(a) + sqrt(b)).

    Corrects favourite-longshot bias, which matters here because the tails are
    extreme and a desperate rival may hail-Mary one.
    """
    pi_h, pi_a = 1.0 / home_odds, 1.0 / away_odds
    a = pi_h / (1.0 - pi_h)
    b = pi_a / (1.0 - pi_a)
    ra, rb = math.sqrt(a), math.sqrt(b)
    return ra / (ra + rb), rb / (ra + rb)


def devig_shin(home_odds: float, away_odds: float, tol: float = 1e-12) -> Tuple[float, float]:
    """Shin (1993) devig, by bisection on the insider-trading fraction z in [0, 0.2].

    p_i = (sqrt(z^2 + 4(1-z) pi_i^2 / B) - z) / (2(1-z)); z solves sum(p_i) = 1.
    """
    pi_h, pi_a = 1.0 / home_odds, 1.0 / away_odds
    b = pi_h + pi_a

    def total(z: float) -> float:
        s = 0.0
        for pi in (pi_h, pi_a):
            s += (math.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / b) - z) / (2.0 * (1.0 - z))
        return s

    lo, hi = 0.0, 0.2
    if total(lo) <= 1.0:
        return devig_proportional(home_odds, away_odds)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    z = 0.5 * (lo + hi)

    def p_of(pi: float) -> float:
        return (math.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / b) - z) / (2.0 * (1.0 - z))

    ph, pa = p_of(pi_h), p_of(pi_a)
    s = ph + pa
    return ph / s, pa / s


DEVIG_METHODS: Dict[str, Callable[[float, float], Tuple[float, float]]] = {
    "proportional": devig_proportional,
    "odds_ratio": devig_odds_ratio,
    "shin": devig_shin,
}


def favourite_prob(game: Game, method: str) -> Tuple[float, str]:
    """Return (probability the favourite wins, favourite team name).

    A per-game manual override beats every devig method.
    """
    if game.p_home_override is not None:
        p_home = game.p_home_override
    else:
        p_home, _ = DEVIG_METHODS[method](game.home_odds, game.away_odds)
    if game.home_odds <= game.away_odds:
        return p_home, game.home
    return 1.0 - p_home, game.away


def underdog_name(game: Game) -> str:
    return game.away if game.home_odds <= game.away_odds else game.home


# --------------------------------------------------------------------------------
# Margin countback model (section 7)
#
# Two margin games remain. Actual margin M ~ Normal(line, SIGMA_MARGIN); each
# tipster's margin tip ~ Normal(line, TAU_TIP); error contribution is |M - tip|.
#
# Every tipster's error is measured against the SAME actual margin, so countback
# outcomes are correlated across rivals. Simulating M directly captures the joint
# distribution; multiplying pairwise probabilities together would overstate safety.
# --------------------------------------------------------------------------------

class CountbackModel:
    """Monte Carlo over the two remaining margin games.

    Exposes `subset_prob(frozenset_of_rival_indices)` = P(my final cumulative margin
    error is lower than every rival in that set).
    """

    def __init__(
        self,
        me: Tipster,
        rivals: Sequence[Tipster],
        margin_lines: Sequence[float],
        my_margin_tips: Optional[Sequence[float]] = None,
        tau: float = TAU_TIP,
        sigma: float = SIGMA_MARGIN,
        n_sims: int = 100_000,
        seed: int = 20260814,
    ) -> None:
        self.me = me
        self.rivals = list(rivals)
        self.n = len(self.rivals)
        self.margin_lines = list(margin_lines)
        self.my_margin_tips = (
            list(my_margin_tips) if my_margin_tips is not None else list(margin_lines)
        )
        self.tau, self.sigma, self.n_sims, self.seed = tau, sigma, n_sims, seed
        self._counts = [0] * (1 << self.n) if self.n else [0]
        self._pairwise = [0.0] * self.n
        self._simulate()

    def _simulate(self) -> None:
        rng = random.Random(self.seed)
        n = self.n
        counts = [0] * (1 << n)
        pair = [0] * n
        base_me = float(self.me.margin_error)
        base_rivals = [float(r.margin_error) for r in self.rivals]
        gauss = rng.gauss

        for _ in range(self.n_sims):
            err_me = base_me
            err = list(base_rivals)
            for gi, line in enumerate(self.margin_lines):
                actual = gauss(line, self.sigma)
                err_me += abs(actual - self.my_margin_tips[gi])
                for j in range(n):
                    err[j] += abs(actual - gauss(line, self.tau))
            mask = 0
            for j in range(n):
                if err_me < err[j]:
                    mask |= 1 << j
                    pair[j] += 1
            counts[mask] += 1

        # P(I beat every rival in S) = fraction of sims whose mask is a superset of S.
        total = [0] * (1 << n)
        for mask in range(1 << n):
            c = counts[mask]
            if not c:
                continue
            sub = mask
            while True:
                total[sub] += c
                if sub == 0:
                    break
                sub = (sub - 1) & mask
        self._counts = total
        self._pairwise = [p / self.n_sims for p in pair]

    def subset_prob(self, indices: Tuple[int, ...]) -> float:
        if not indices:
            return 1.0
        mask = 0
        for i in indices:
            mask |= 1 << i
        return self._counts[mask] / self.n_sims

    def pairwise(self, index: int) -> float:
        return self._pairwise[index]


def pairwise_countback_static(a: Tipster, b: Tipster) -> float:
    """P(a beats b on countback), ignoring the two remaining margin games.

    Used only to model how RIVALS believe their own countbacks will resolve. My own
    countback always goes through the full CountbackModel.
    """
    if a.margin_error < b.margin_error:
        return 1.0
    if a.margin_error > b.margin_error:
        return 0.0
    return 0.5


# --------------------------------------------------------------------------------
# Level-0 backward induction over (t, delta)
#
# delta = net points versus an all-favourites baseline. Tipping the favourite leaves
# delta unchanged; tipping the dog moves it +1 if the dog wins, -1 if it loses.
# --------------------------------------------------------------------------------

def solve_level0(
    p_fav: Sequence[float],
    terminal: Callable[[int], float],
    clamp: int = DELTA_CLAMP,
) -> Tuple[List[List[float]], List[List[str]]]:
    """Exact DP. Returns (values, policy); index delta as [t][delta + clamp]."""
    n = len(p_fav)
    size = 2 * clamp + 1
    values = [[0.0] * size for _ in range(n + 1)]
    policy = [["F"] * size for _ in range(n)]

    for d in range(-clamp, clamp + 1):
        values[n][d + clamp] = terminal(d)

    for t in range(n - 1, -1, -1):
        p = p_fav[t]
        nxt = values[t + 1]
        for d in range(-clamp, clamp + 1):
            i = d + clamp
            v_fav = nxt[i]
            lo = max(-clamp, d - 1) + clamp
            hi = min(clamp, d + 1) + clamp
            v_dog = p * nxt[lo] + (1.0 - p) * nxt[hi]
            if v_dog > v_fav + 1e-12:
                values[t][i], policy[t][i] = v_dog, "D"
            else:
                values[t][i], policy[t][i] = v_fav, "F"
    return values, policy


def rival_terminal(
    rival: Tipster,
    others: Sequence[Tipster],
    my_name: Optional[str] = None,
    my_assumed_gain: int = 0,
) -> Callable[[int], float]:
    """Rival's own win condition, under their level-0 belief that everyone else
    tips favourites (so everyone else's points are frozen at today's values).

    Captures the ADRIANOartini asymmetry: his 646 margin error loses the countback
    to Ryan's 628, so he needs +3 outright where Markash (574) needs only +2.

    `my_assumed_gain` inflates MY points in the rival's belief. At 0 the rival is
    purely level-0 and satisfices: once their delta is high enough to beat a frozen
    field they stop deviating, which flatters me. Raising it models a chaser who
    anticipates that the tipster one point off the lead is also chasing, and so keeps
    pushing. Reported as a sensitivity rather than picked, because it is a belief
    about rivals that no input file can settle.
    """
    def terminal(delta: int) -> float:
        final = rival.points + delta
        value = 1.0
        for o in others:
            target = o.points + (my_assumed_gain if (my_name and o.name == my_name) else 0)
            if final < target:
                return 0.0
            if final == target:
                value *= pairwise_countback_static(rival, o)
        return value
    return terminal


# --------------------------------------------------------------------------------
# The joint DP: me against grouped rivals
# --------------------------------------------------------------------------------

@dataclass
class RivalGroup:
    """Rivals sharing an identical policy share a delta, so they collapse to one
    DP dimension. Ryan (always-favourite) has delta identically zero and needs no
    dimension at all."""
    members: List[int]              # indices into the rivals list
    policy: Optional[List[List[str]]]   # None => always favourite


@dataclass
class Solution:
    p_win: float
    action: str                     # "F" or "D"
    p_win_favourite: float
    p_win_dog: float
    groups: List[RivalGroup]
    rival_names: List[str]
    action_at: Optional[Callable[[int, int, Tuple[int, ...]], Tuple[str, float, float]]] = None


def build_rival_groups(
    rivals: Sequence[Tipster],
    all_tipsters: Sequence[Tipster],
    p_fav: Sequence[float],
    leader_tips_favourites: bool = True,
    my_assumed_gain: int = 0,
) -> List[RivalGroup]:
    """Solve each rival's level-0 policy, then group rivals whose policies match.

    Per user instruction: the leader tips favourites, the chasers do not.
    """
    leader_points = max(t.points for t in all_tipsters)
    my_name = next((t.name for t in all_tipsters if t.is_me), None)
    signature_to_group: Dict[Tuple, RivalGroup] = {}
    groups: List[RivalGroup] = []

    for idx, rival in enumerate(rivals):
        is_leader = rival.points == leader_points
        if is_leader and leader_tips_favourites:
            signature: Tuple = ("always_favourite",)
            policy = None
        else:
            others = [t for t in all_tipsters if t.name != rival.name]
            _, policy = solve_level0(
                p_fav, rival_terminal(rival, others, my_name, my_assumed_gain)
            )
            signature = tuple(tuple(row) for row in policy)

        existing = signature_to_group.get(signature)
        if existing is not None:
            existing.members.append(idx)
        else:
            group = RivalGroup(members=[idx], policy=policy)
            signature_to_group[signature] = group
            groups.append(group)
    return groups


def solve_joint(
    me: Tipster,
    rivals: Sequence[Tipster],
    p_fav: Sequence[float],
    groups: Sequence[RivalGroup],
    countback: CountbackModel,
    clamp: int = DELTA_CLAMP,
) -> Solution:
    """Exact backward induction over (t, delta_me, delta_per_group).

    Rival policies do not respond to my choices, so their deltas can be carried as
    exact extra dimensions rather than approximated.
    """
    n_games = len(p_fav)
    gaps = [r.points - me.points for r in rivals]
    group_of = {}
    for gi, group in enumerate(groups):
        for member in group.members:
            group_of[member] = gi
    n_groups = len(groups)

    def terminal(delta_me: int, group_deltas: Tuple[int, ...]) -> float:
        tied: List[int] = []
        for j, gap in enumerate(gaps):
            diff = gap + group_deltas[group_of[j]] - delta_me
            if diff > 0:
                return 0.0            # beaten outright
            if diff == 0:
                tied.append(j)
        return countback.subset_prob(tuple(tied))

    def clamp_d(d: int) -> int:
        return max(-clamp, min(clamp, d))

    @lru_cache(maxsize=None)
    def value(t: int, delta_me: int, group_deltas: Tuple[int, ...]) -> float:
        if t == n_games:
            return terminal(delta_me, group_deltas)

        remaining = n_games - t
        # Elimination prune: I gain at most 1 point per game on any rival.
        worst = max(
            gaps[j] + group_deltas[group_of[j]] - delta_me for j in range(len(gaps))
        )
        if worst - remaining > 0:
            return 0.0

        p = p_fav[t]
        # Each group's action at this node comes from its own precomputed policy.
        group_actions = []
        for gi, group in enumerate(groups):
            if group.policy is None:
                group_actions.append("F")
            else:
                group_actions.append(group.policy[t][clamp_d(group_deltas[gi]) + clamp])

        def step(my_action: str, fav_won: bool) -> Tuple[int, Tuple[int, ...]]:
            move = (1 if not fav_won else -1)
            nd_me = clamp_d(delta_me + (move if my_action == "D" else 0))
            nd_groups = tuple(
                clamp_d(group_deltas[gi] + (move if group_actions[gi] == "D" else 0))
                for gi in range(n_groups)
            )
            return nd_me, nd_groups

        best = -1.0
        for my_action in ("F", "D"):
            fav_me, fav_groups = step(my_action, True)
            dog_me, dog_groups = step(my_action, False)
            v = p * value(t + 1, fav_me, fav_groups) + (1.0 - p) * value(
                t + 1, dog_me, dog_groups
            )
            if v > best:
                best = v
        return best

    # Evaluate both options for the next game explicitly.
    start_groups = tuple(0 for _ in range(n_groups))
    p = p_fav[0]
    group_actions0 = []
    for group in groups:
        group_actions0.append("F" if group.policy is None else group.policy[0][clamp])

    def option_value(my_action: str) -> float:
        total = 0.0
        for fav_won, prob in ((True, p), (False, 1.0 - p)):
            move = -1 if fav_won else 1
            nd_me = clamp_d(0 + (move if my_action == "D" else 0))
            nd_groups = tuple(
                clamp_d(0 + (move if group_actions0[gi] == "D" else 0))
                for gi in range(n_groups)
            )
            total += prob * value(1, nd_me, nd_groups)
        return total

    v_fav, v_dog = option_value("F"), option_value("D")
    action = "D" if v_dog > v_fav + 1e-12 else "F"

    def action_at(t: int, delta_me: int, group_deltas: Tuple[int, ...]):
        """Optimal action and both option values at an arbitrary node.

        Lets --explain print a contingency table you can read at the ground without
        re-running the solver.
        """
        if t >= n_games:
            return "F", 0.0, 0.0
        p_t = p_fav[t]
        acts = [
            "F" if g.policy is None else g.policy[t][clamp_d(group_deltas[gi]) + clamp]
            for gi, g in enumerate(groups)
        ]
        out = {}
        for my_action in ("F", "D"):
            total = 0.0
            for fav_won, prob in ((True, p_t), (False, 1.0 - p_t)):
                move = -1 if fav_won else 1
                nd_me = clamp_d(delta_me + (move if my_action == "D" else 0))
                nd_groups = tuple(
                    clamp_d(group_deltas[gi] + (move if acts[gi] == "D" else 0))
                    for gi in range(n_groups)
                )
                total += prob * value(t + 1, nd_me, nd_groups)
            out[my_action] = total
        best = "D" if out["D"] > out["F"] + 1e-12 else "F"
        return best, out["F"], out["D"]

    return Solution(
        p_win=max(v_fav, v_dog),
        action=action,
        p_win_favourite=v_fav,
        p_win_dog=v_dog,
        groups=list(groups),
        rival_names=[r.name for r in rivals],
        action_at=action_at,
    )


def evaluate_fixed_policy(
    me: Tipster,
    rivals: Sequence[Tipster],
    p_fav: Sequence[float],
    groups: Sequence[RivalGroup],
    countback: CountbackModel,
    my_actions: Callable[[int, int], str],
    clamp: int = DELTA_CLAMP,
) -> float:
    """P(win) for a fixed policy of mine, same rival model. Used for baselines."""
    n_games = len(p_fav)
    gaps = [r.points - me.points for r in rivals]
    group_of = {}
    for gi, group in enumerate(groups):
        for member in group.members:
            group_of[member] = gi
    n_groups = len(groups)

    def clamp_d(d: int) -> int:
        return max(-clamp, min(clamp, d))

    @lru_cache(maxsize=None)
    def value(t: int, delta_me: int, group_deltas: Tuple[int, ...]) -> float:
        if t == n_games:
            tied: List[int] = []
            for j, gap in enumerate(gaps):
                diff = gap + group_deltas[group_of[j]] - delta_me
                if diff > 0:
                    return 0.0
                if diff == 0:
                    tied.append(j)
            return countback.subset_prob(tuple(tied))

        p = p_fav[t]
        my_action = my_actions(t, delta_me)
        group_actions = [
            "F" if g.policy is None else g.policy[t][clamp_d(group_deltas[gi]) + clamp]
            for gi, g in enumerate(groups)
        ]
        total = 0.0
        for fav_won, prob in ((True, p), (False, 1.0 - p)):
            move = -1 if fav_won else 1
            nd_me = clamp_d(delta_me + (move if my_action == "D" else 0))
            nd_groups = tuple(
                clamp_d(group_deltas[gi] + (move if group_actions[gi] == "D" else 0))
                for gi in range(n_groups)
            )
            total += prob * value(t + 1, nd_me, nd_groups)
        return total

    return value(0, 0, tuple(0 for _ in range(n_groups)))


# --------------------------------------------------------------------------------
# Input / output
# --------------------------------------------------------------------------------

TEMPLATE_LEADERBOARD = [
    ["name", "points", "margin_error", "is_me"],
    ["Ryan Board", "147", "628", "0"],
    ["Jake Turner", "146", "573", "1"],
    ["NRL > AFL", "146", "677", "0"],
    ["Markash", "145", "574", "0"],
    ["DeanLFC", "145", "578", "0"],
    ["Mikefooty", "145", "587", "0"],
    ["ADRIANOartini", "145", "646", "0"],
]

TEMPLATE_FIXTURES_HEADER = [
    "game_id", "round", "lock_local", "home", "away",
    "home_odds", "away_odds", "line_fav", "is_margin_game", "p_home_override",
]


def write_template(paths: SetPaths) -> None:
    os.makedirs(os.path.dirname(paths.leaderboard), exist_ok=True)
    with open(paths.leaderboard, "w", newline="") as fh:
        csv.writer(fh).writerows(TEMPLATE_LEADERBOARD)

    rows = [TEMPLATE_FIXTURES_HEADER]
    for rnd in ("R23", "R24"):
        for g in range(1, 10):
            rows.append([
                "%sG%d" % (rnd, g), rnd, "", "", "", "", "", "",
                "1" if g == 1 else "0", "",
            ])
    with open(paths.fixtures, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    print("Wrote %s" % paths.leaderboard)
    print("Wrote %s" % paths.fixtures)
    print()
    print("Fill in fixtures.csv, one row per remaining game, IN LOCK ORDER:")
    print("  home / away        team names")
    print("  home_odds/away_odds  decimal head-to-head odds")
    print("  line_fav           favourite's expected winning margin, POSITIVE")
    print("  is_margin_game     1 for the first game of each round, else 0")
    print("  p_home_override    optional; your own model's P(home win), beats the devig")
    print()
    print("Delete rows for games already played. Then: python3 tipping.py --recommend --set %s"
          % paths.name)


def copy_set(source: SetPaths, dest: SetPaths, force: bool = False,
             confirm: Callable[[str], str] = input) -> List[str]:
    """Clone `source` over `dest`, prompting before destroying existing edits.

    Returns the paths written, or an empty list if the user declined. `confirm`
    is injected so the prompt can be exercised in tests.
    """
    if source.name == dest.name:
        raise InputError("cannot copy set %r onto itself" % source.name)
    for path in (source.leaderboard, source.fixtures):
        if not os.path.exists(path):
            raise InputError(
                "source set %r is incomplete: %s not found" % (source.name, path)
            )

    dest_dir = os.path.dirname(dest.leaderboard)
    existing = [p for p in (dest.leaderboard, dest.fixtures) if os.path.exists(p)]
    if existing and not force:
        answer = confirm("Overwrite %s from set %r? [y/N] " % (dest_dir, source.name))
        if answer.strip().lower() not in ("y", "yes"):
            return []

    os.makedirs(dest_dir, exist_ok=True)
    shutil.copyfile(source.leaderboard, dest.leaderboard)
    shutil.copyfile(source.fixtures, dest.fixtures)
    return [dest.leaderboard, dest.fixtures]


def _num(value: str, field: str, row: int, path: str, cast=float):
    text = (value or "").strip()
    if text == "":
        raise InputError("%s row %d: column '%s' is empty" % (path, row, field))
    try:
        return cast(text)
    except ValueError:
        raise InputError(
            "%s row %d: column '%s' has non-numeric value %r" % (path, row, field, text)
        )


def load_leaderboard(path: str = LEADERBOARD_CSV) -> Tuple[Tipster, List[Tipster]]:
    if not os.path.exists(path):
        raise InputError("%s not found. Run: python3 tipping.py --make-template" % path)
    tipsters: List[Tipster] = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            name = (row.get("name") or "").strip()
            if not name:
                continue           # blank rows are allowed; rows 8+ may be filled later
            tipsters.append(Tipster(
                name=name,
                points=int(_num(row.get("points", ""), "points", i, path, float)),
                margin_error=int(_num(row.get("margin_error", ""), "margin_error", i, path, float)),
                is_me=(row.get("is_me") or "0").strip() in ("1", "true", "TRUE", "yes"),
            ))
    if not tipsters:
        raise InputError("%s contains no tipsters" % path)
    me = [t for t in tipsters if t.is_me]
    if len(me) != 1:
        raise InputError(
            "%s must have exactly one row with is_me=1 (found %d)" % (path, len(me))
        )
    return me[0], [t for t in tipsters if not t.is_me]


def load_fixtures(path: str = FIXTURES_CSV) -> List[Game]:
    if not os.path.exists(path):
        raise InputError("%s not found. Run: python3 tipping.py --make-template" % path)
    games: List[Game] = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            gid = (row.get("game_id") or "").strip()
            home = (row.get("home") or "").strip()
            away = (row.get("away") or "").strip()
            if not gid or (not home and not away):
                continue            # unfilled template row
            ho = _num(row.get("home_odds", ""), "home_odds", i, path)
            ao = _num(row.get("away_odds", ""), "away_odds", i, path)
            if ho <= 1.0 or ao <= 1.0:
                raise InputError(
                    "%s row %d: decimal odds must exceed 1.0 (got %s / %s)" % (path, i, ho, ao)
                )
            line_text = (row.get("line_fav") or "").strip()
            override_text = (row.get("p_home_override") or "").strip()
            override = float(override_text) if override_text else None
            if override is not None and not (0.0 < override < 1.0):
                raise InputError(
                    "%s row %d: p_home_override must be strictly between 0 and 1" % (path, i)
                )
            games.append(Game(
                game_id=gid,
                round=(row.get("round") or "").strip(),
                lock_local=(row.get("lock_local") or "").strip(),
                home=home, away=away,
                home_odds=ho, away_odds=ao,
                line_fav=float(line_text) if line_text else None,
                is_margin_game=(row.get("is_margin_game") or "0").strip() in ("1", "true", "TRUE", "yes"),
                p_home_override=override,
            ))
    if not games:
        raise InputError(
            "%s has no completed rows. Fill in the fixtures before running --recommend." % path
        )
    return games


# --------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------

def pct(x: float) -> str:
    return "%6.2f%%" % (100.0 * x)


def pct_mc(x: float, n_sims: int) -> str:
    """Format a Monte Carlo probability without claiming a certainty the sample
    cannot support. Zero counterexamples in N draws is not proof of 100%."""
    if x >= 1.0:
        return ">%.3f%%" % (100.0 * (1.0 - 1.0 / n_sims))
    if x <= 0.0:
        return "<%.3f%%" % (100.0 / n_sims)
    return pct(x)


def rule(char: str = "-", width: int = 78) -> str:
    return char * width


def report(
    me: Tipster,
    rivals: List[Tipster],
    games: List[Game],
    method: str,
    explain: bool,
    n_sims: int,
    seed: int,
    tau: float,
) -> Dict[str, object]:
    p_fav = []
    fav_names = []
    for g in games:
        p, fav = favourite_prob(g, method)
        p_fav.append(p)
        fav_names.append(fav)

    margin_indices = [i for i, g in enumerate(games) if g.is_margin_game]
    margin_lines = []
    for i in margin_indices:
        line = games[i].line_fav
        if line is None:
            raise InputError(
                "%s is flagged as a margin game but has no line_fav" % games[i].game_id
            )
        margin_lines.append(line)

    countback = CountbackModel(
        me, rivals, margin_lines, tau=tau, n_sims=n_sims, seed=seed
    )
    groups = build_rival_groups(rivals, [me] + rivals, p_fav)
    solution = solve_joint(me, rivals, p_fav, groups, countback)

    g0 = games[0]
    fav0, dog0 = fav_names[0], underdog_name(g0)

    print()
    print(rule("="))
    print("AFL TIPPING -- NEXT DECISION")
    print(rule("="))
    print("Game        : %s  %s v %s" % (g0.game_id, g0.home, g0.away))
    if g0.lock_local:
        print("Locks       : %s" % g0.lock_local)
    print("Favourite   : %-22s  P(win) = %s   [%s devig]" % (fav0, pct(p_fav[0]), method))
    print("Underdog    : %-22s  P(win) = %s" % (dog0, pct(1.0 - p_fav[0])))
    print()
    print("  Tip the FAVOURITE (%-18s)  ->  P(win comp) = %s" % (fav0, pct(solution.p_win_favourite)))
    print("  Tip the UNDERDOG  (%-18s)  ->  P(win comp) = %s" % (dog0, pct(solution.p_win_dog)))
    print()
    edge = abs(solution.p_win_dog - solution.p_win_favourite)
    chosen = dog0 if solution.action == "D" else fav0
    print("  >>> RECOMMENDATION: tip %s   (edge %s)" % (chosen.upper(), pct(edge)))
    print()

    # ---- Why -------------------------------------------------------------------
    print(rule())
    print("WHY")
    print(rule())
    gaps = {r.name: r.points - me.points for r in rivals}
    leader = max(rivals, key=lambda r: r.points)
    print("You trail %s by %d point(s) and hold the lowest margin error in the field" %
          (leader.name, gaps[leader.name]))
    print("(%d v %d), so a tie for first is a win for you against every current rival." %
          (me.margin_error, leader.margin_error))
    print()
    print("Every rival's score is (their points + F), where F is however many favourites")
    print("win -- the same F for everyone. F cancels, so only DIFFERENTIALS matter:")
    print("tipping the favourite alongside everyone else cannot change your position.")
    print()
    print("Countback safety, from the line-derived margin model (tau = %.0f, ASSUMED):" % tau)
    for j, r in enumerate(rivals):
        print("    vs %-16s margin gap %+5d   P(you win countback) = %s" %
              (r.name, r.margin_error - me.margin_error,
               pct_mc(countback.pairwise(j), n_sims)))
    print()
    print("Rival model -- leader tips favourites, chasers deviate to close the gap:")
    for gi, group in enumerate(groups):
        names = ", ".join(rivals[m].name for m in group.members)
        if group.policy is None:
            print("    group %d: %-46s always favourite" % (gi, names))
        else:
            devs = sum(1 for t in range(len(p_fav)) if group.policy[t][DELTA_CLAMP] == "D")
            print("    group %d: %-46s deviates in %d game(s) on the level path" %
                  (gi, names, devs))
    print()

    # ---- Contingency table -----------------------------------------------------
    if explain and solution.action_at is not None:
        print(rule())
        print("CONTINGENCY TABLE  (next %d games)" % min(3, len(games)))
        print(rule())
        print("Read this at the ground. 'delta' is your net points versus tipping every")
        print("favourite: +1 for each dog you took that won, -1 for each that lost.")
        print("Rival deltas are held on their level path, which is where they start.")
        print()
        zero = tuple(0 for _ in groups)
        for t in range(min(3, len(games))):
            g = games[t]
            fav_t, dog_t = fav_names[t], underdog_name(g)
            print("  %s  %s v %s   favourite %s (%s)" %
                  (g.game_id, g.home, g.away, fav_t, pct(p_fav[t])))
            for d in range(-2, 3):
                best, v_f, v_d = solution.action_at(t, d, zero)
                label = dog_t if best == "D" else fav_t
                print("      delta %+d  ->  tip %-20s  fav %s   dog %s" %
                      (d, label, pct(v_f), pct(v_d)))
            print()

    # ---- Baselines -------------------------------------------------------------
    print(rule())
    print("BASELINE COMPARISON  (same rival model throughout)")
    print(rule())
    baselines = {
        "always favourite": lambda t, d: "F",
        "deviate immediately": lambda t, d: "D" if t == 0 else "F",
        "deviate on the last game": lambda t, d: "D" if t == len(p_fav) - 1 else "F",
    }
    rows = []
    for label, fn in baselines.items():
        v = evaluate_fixed_policy(me, rivals, p_fav, groups, countback, fn)
        rows.append((label, v))
        print("    %-28s P(win) = %s" % (label, pct(v)))
    print("    %-28s P(win) = %s   <-- optimal" % ("this engine", pct(solution.p_win)))
    print()

    # ---- Chaser model sensitivity ----------------------------------------------
    # The pure level-0 chaser believes the field is frozen, so once their delta is
    # high enough to win they stop deviating. A real chaser watching you pull clear
    # would keep pushing. This is a belief about rivals that no input file settles,
    # so both are reported rather than one being picked.
    print(rule())
    print("CHASER MODEL SENSITIVITY")
    print(rule())
    chaser_rows = []
    for label, gain in (("satisficing (pure level-0)", 0),
                        ("relentless (expects you at +1)", 1),
                        ("relentless (expects you at +2)", 2)):
        grp = build_rival_groups(rivals, [me] + rivals, p_fav, my_assumed_gain=gain)
        sol = solve_joint(me, rivals, p_fav, grp, countback)
        chaser_rows.append((label, gain, sol.p_win, sol.action))
        marker = "   <-- headline" if gain == 0 else ""
        print("    %-32s P(win) = %s   tip %s%s" %
              (label, pct(sol.p_win), dog0 if sol.action == "D" else fav0, marker))
    spread = max(r[2] for r in chaser_rows) - min(r[2] for r in chaser_rows)
    print()
    print("    Spread across chaser models: %s" % pct(spread))
    if len({r[3] for r in chaser_rows}) > 1:
        print("    *** WARNING: the recommended action FLIPS between chaser models. ***")
    print()

    # ---- Devig sensitivity -----------------------------------------------------
    print(rule())
    print("DEVIG SENSITIVITY")
    print(rule())
    actions = {}
    sens_rows = []
    for m in ("proportional", "odds_ratio", "shin"):
        pf = [favourite_prob(g, m)[0] for g in games]
        grp = build_rival_groups(rivals, [me] + rivals, pf)
        sol = solve_joint(me, rivals, pf, grp, countback)
        actions[m] = sol.action
        sens_rows.append((m, pf[0], sol.p_win, sol.action))
        print("    %-14s P(fav game 1) = %s   P(win) = %s   tip %s" %
              (m, pct(pf[0]), pct(sol.p_win), dog0 if sol.action == "D" else fav0))
    if len(set(actions.values())) > 1:
        print()
        print("    *** WARNING: the recommended action FLIPS between devig methods. ***")
        print("    *** Treat this decision as genuinely marginal. ***")
    print()

    # ---- Margin tip ------------------------------------------------------------
    print(rule())
    if margin_indices:
        print("MARGIN TIP  (%s)" % games[margin_indices[0]].game_id)
    else:
        print("MARGIN TIP  (none remaining)")
    print(rule())
    margin_rows = []
    if margin_indices:
        line0 = margin_lines[0]
        best_tip, best_val = None, -1.0
        for offset in range(-20, 21, 5):
            tips = list(margin_lines)
            tips[0] = line0 + offset
            cb = CountbackModel(me, rivals, margin_lines, my_margin_tips=tips,
                                tau=tau, n_sims=max(20_000, n_sims // 4), seed=seed)
            sol = solve_joint(me, rivals, p_fav, groups, cb)
            margin_rows.append((line0 + offset, offset, sol.p_win))
            if sol.p_win > best_val:
                best_tip, best_val = line0 + offset, sol.p_win
        on_line = next(v for tip, offset, v in margin_rows if offset == 0)
        # Only call a winner if it beats the line by more than the sweep's own noise.
        MEANINGFUL = 0.0025
        print("    Line for %s is %+.0f. Win probability by your margin tip:" %
              (games[margin_indices[0]].game_id, line0))
        for tip, offset, v in margin_rows:
            marker = "  <-- best" if tip == best_tip else ""
            print("      tip %+6.0f  (line %+3d)   P(win) = %s%s" % (tip, offset, pct(v), marker))
        print()
        if best_val - on_line > MEANINGFUL:
            print("    >>> RECOMMENDATION: tip %+.0f  (beats the line by %s)" %
                  (best_tip, pct(best_val - on_line)))
        else:
            print("    >>> RECOMMENDATION: tip the line, %+.0f" % line0)
            print("    The curve is flat to within %s across the whole sweep, so the" %
                  pct(best_val - on_line))
            print("    apparent 'best' is inside the Monte Carlo noise. Tipping the line")
            print("    minimises expected error (the median minimises E|M-m|), and your")
            print("    margin lead is your single most valuable asset -- protect it.")
    print()

    print(rule("="))
    print("ASSUMPTIONS  (see docs/superpowers/specs/ for the full list)")
    print(rule("="))
    print("  * tau = %.0f is ASSUMED, not fitted. Fill past rounds' margin tips to measure it." % tau)
    print("  * Rivals are level-0: they play the field, they do not respond to you.")
    print("  * Leader tips favourites; chasers deviate per their own best response.")
    print("  * Seed = %d, %d countback sims. Deterministic given this seed." % (seed, n_sims))
    print(rule("="))
    print()

    return {
        "solution": solution,
        "p_fav": p_fav,
        "fav_names": fav_names,
        "baselines": rows,
        "sensitivity": sens_rows,
        "margin_rows": margin_rows,
        "countback": countback,
        "groups": groups,
    }


def write_csv_outputs(me: Tipster, rivals: List[Tipster], games: List[Game],
                      result: Dict[str, object], method: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "recommendation.csv")
    sol: Solution = result["solution"]           # type: ignore[assignment]
    p_fav: List[float] = result["p_fav"]          # type: ignore[assignment]
    fav_names: List[str] = result["fav_names"]    # type: ignore[assignment]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value", "detail"])
        w.writerow(["next", "game_id", games[0].game_id,
                    "%s v %s" % (games[0].home, games[0].away)])
        w.writerow(["next", "lock_local", games[0].lock_local, ""])
        w.writerow(["next", "favourite", fav_names[0], "%.6f" % p_fav[0]])
        w.writerow(["next", "p_win_if_favourite", "%.6f" % sol.p_win_favourite, ""])
        w.writerow(["next", "p_win_if_underdog", "%.6f" % sol.p_win_dog, ""])
        w.writerow(["next", "recommended",
                    fav_names[0] if sol.action == "F" else underdog_name(games[0]),
                    "edge %.6f" % abs(sol.p_win_dog - sol.p_win_favourite)])
        for label, v in result["baselines"]:      # type: ignore[union-attr]
            w.writerow(["baseline", label, "%.6f" % v, ""])
        w.writerow(["baseline", "this engine", "%.6f" % sol.p_win, "optimal"])
        for m, pf0, v, act in result["sensitivity"]:   # type: ignore[union-attr]
            w.writerow(["devig", m, "%.6f" % v, "p_fav_game1=%.6f action=%s" % (pf0, act)])
        for tip, offset, v in result["margin_rows"]:   # type: ignore[union-attr]
            w.writerow(["margin", "%+.0f" % tip, "%.6f" % v, "line%+d" % offset])
        cb: CountbackModel = result["countback"]  # type: ignore[assignment]
        for j, r in enumerate(rivals):
            w.writerow(["countback", r.name, "%.6f" % cb.pairwise(j),
                        "margin gap %+d" % (r.margin_error - me.margin_error)])
        for i, g in enumerate(games):
            w.writerow(["game", g.game_id, "%.6f" % p_fav[i],
                        "favourite=%s devig=%s" % (fav_names[i], method)])
    return path


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-template", action="store_true",
                        help="write blank inputs/*.csv to fill in")
    parser.add_argument("--recommend", action="store_true",
                        help="solve and print the next decision")
    parser.add_argument("--explain", action="store_true",
                        help="dump the reasoning for the next few games")
    parser.add_argument("--fixtures", default=FIXTURES_CSV, help="path to fixtures CSV")
    parser.add_argument("--leaderboard", default=LEADERBOARD_CSV, help="path to leaderboard CSV")
    parser.add_argument("--devig", default="odds_ratio",
                        choices=sorted(DEVIG_METHODS), help="devig method (default odds_ratio)")
    parser.add_argument("--tau", type=float, default=TAU_TIP,
                        help="SD of rivals' margin tips around the line (default %.0f)" % TAU_TIP)
    parser.add_argument("--sims", type=int, default=100_000, help="countback Monte Carlo draws")
    parser.add_argument("--seed", type=int, default=20260814, help="RNG seed")
    args = parser.parse_args(argv)

    if args.make_template:
        write_template()
        return 0
    if not args.recommend:
        parser.print_help()
        return 0

    try:
        me, rivals = load_leaderboard(args.leaderboard)
        games = load_fixtures(args.fixtures)
    except InputError as exc:
        print("INPUT ERROR: %s" % exc, file=sys.stderr)
        return 2

    result = report(me, rivals, games, args.devig, args.explain,
                    args.sims, args.seed, args.tau)
    path = write_csv_outputs(me, rivals, games, result, args.devig)
    print("Wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
