#!/usr/bin/env python3
"""Tests for the AFL tipping engine. Pure stdlib unittest, no third-party deps.

Run:  python3 test_tipping.py
"""

import math
import random
import unittest

import tipping as T


class TestDevig(unittest.TestCase):
    """Section 3 of the brief: the closed forms, and the sanity check it specifies."""

    def test_brief_sanity_check_1_02_vs_15_00(self):
        # The brief asserts: proportional gives the longshot 6.4%, odds-ratio 3.6%.
        _, pa_prop = T.devig_proportional(1.02, 15.00)
        _, pa_or = T.devig_odds_ratio(1.02, 15.00)
        self.assertAlmostEqual(pa_prop, 0.063670, places=5)
        self.assertAlmostEqual(pa_or, 0.036420, places=5)
        self.assertEqual(round(100 * pa_prop, 1), 6.4)
        self.assertEqual(round(100 * pa_or, 1), 3.6)

    def test_all_methods_normalise(self):
        for oh, ao in ((1.02, 15.00), (1.90, 1.95), (1.20, 4.80), (2.50, 1.55)):
            for name, fn in T.DEVIG_METHODS.items():
                ph, pa = fn(oh, ao)
                self.assertAlmostEqual(ph + pa, 1.0, places=9, msg=name)
                self.assertTrue(0.0 < ph < 1.0, msg=name)

    def test_odds_ratio_shortens_the_longshot(self):
        # Favourite-longshot bias correction: the odds-ratio devig must assign the
        # longshot LESS probability than proportional does.
        for oh, ao in ((1.02, 15.00), (1.10, 8.00), (1.20, 4.80)):
            _, pa_prop = T.devig_proportional(oh, ao)
            _, pa_or = T.devig_odds_ratio(oh, ao)
            self.assertLess(pa_or, pa_prop)

    def test_shin_lies_between_the_two(self):
        for oh, ao in ((1.02, 15.00), (1.10, 8.00), (1.20, 4.80)):
            _, pa_prop = T.devig_proportional(oh, ao)
            _, pa_or = T.devig_odds_ratio(oh, ao)
            _, pa_shin = T.devig_shin(oh, ao)
            self.assertLessEqual(pa_or - 1e-9, pa_shin)
            self.assertLessEqual(pa_shin, pa_prop + 1e-9)

    def test_no_vig_market_is_unchanged(self):
        ph, pa = T.devig_proportional(2.0, 2.0)
        self.assertAlmostEqual(ph, 0.5, places=9)
        ph, pa = T.devig_odds_ratio(2.0, 2.0)
        self.assertAlmostEqual(ph, 0.5, places=9)


class TestFavouriteIdentification(unittest.TestCase):
    def test_shorter_odds_is_the_favourite(self):
        g = T.Game("R23G1", "R23", "", "Home", "Away", 1.20, 4.80, 25.0, True)
        p, fav = T.favourite_prob(g, "odds_ratio")
        self.assertEqual(fav, "Home")
        self.assertGreater(p, 0.5)
        self.assertEqual(T.underdog_name(g), "Away")

    def test_away_favourite(self):
        g = T.Game("R23G2", "R23", "", "Home", "Away", 4.80, 1.20, 25.0, False)
        p, fav = T.favourite_prob(g, "odds_ratio")
        self.assertEqual(fav, "Away")
        self.assertGreater(p, 0.5)

    def test_manual_override_beats_the_devig(self):
        g = T.Game("R23G3", "R23", "", "Home", "Away", 1.20, 4.80, 25.0, False,
                   p_home_override=0.60)
        p, fav = T.favourite_prob(g, "odds_ratio")
        self.assertEqual(fav, "Home")
        self.assertAlmostEqual(p, 0.60, places=9)


class TestDifferentialTransition(unittest.TestCase):
    """delta = net points versus an all-favourites baseline."""

    def test_tipping_the_favourite_never_moves_delta(self):
        # A one-game DP whose terminal rewards delta == 0 only: tipping the
        # favourite must be optimal and worth exactly 1.
        values, policy = T.solve_level0([0.7], lambda d: 1.0 if d == 0 else 0.0, clamp=5)
        self.assertEqual(policy[0][0 + 5], "F")
        self.assertAlmostEqual(values[0][0 + 5], 1.0, places=9)

    def test_tipping_the_dog_moves_delta_by_the_dog_probability(self):
        # Terminal rewards delta == +1 only. Tipping the dog gets there exactly when
        # the dog wins, so the value must equal the dog's win probability.
        p_fav = 0.7
        values, policy = T.solve_level0([p_fav], lambda d: 1.0 if d == 1 else 0.0, clamp=5)
        self.assertEqual(policy[0][0 + 5], "D")
        self.assertAlmostEqual(values[0][0 + 5], 1.0 - p_fav, places=9)

    def test_value_is_monotone_in_delta(self):
        p_fav = [0.65] * 6
        values, _ = T.solve_level0(p_fav, lambda d: 1.0 if d >= 2 else 0.0, clamp=10)
        row = values[0]
        for i in range(len(row) - 1):
            self.assertLessEqual(row[i] - 1e-12, row[i + 1])

    def test_certain_favourites_make_deviation_worthless(self):
        # If every favourite wins with certainty, deviating can only lose points.
        values, policy = T.solve_level0(
            [1.0] * 5, lambda d: 1.0 if d >= 1 else 0.0, clamp=8
        )
        self.assertAlmostEqual(values[0][0 + 8], 0.0, places=9)
        self.assertTrue(all(policy[t][0 + 8] == "F" for t in range(5)))

    def test_coin_flip_endgame_caps_at_the_dog_probability(self):
        # One game left, need +1: the best you can do is the dog's probability.
        for p in (0.5, 0.6, 0.75):
            values, _ = T.solve_level0([p], lambda d: 1.0 if d >= 1 else 0.0, clamp=4)
            self.assertAlmostEqual(values[0][0 + 4], 1.0 - p, places=9)


ME = T.Tipster("Jake Turner", 146, 573, is_me=True)
RIVALS = [
    T.Tipster("Ryan Board", 147, 628),
    T.Tipster("NRL > AFL", 146, 677),
    T.Tipster("Markash", 145, 574),
    T.Tipster("DeanLFC", 145, 578),
    T.Tipster("Mikefooty", 145, 587),
    T.Tipster("ADRIANOartini", 145, 646),
]


def _flat_fixture(n=18, p=0.7):
    return [p] * n


def _countback(n_sims=4000, tau=T.TAU_TIP):
    return T.CountbackModel(ME, RIVALS, [25.0, 25.0], tau=tau, n_sims=n_sims, seed=7)


class TestCountbackModel(unittest.TestCase):
    def test_probabilities_are_bounded(self):
        cb = _countback()
        for j in range(len(RIVALS)):
            self.assertTrue(0.0 <= cb.pairwise(j) <= 1.0)

    def test_empty_set_is_certain(self):
        self.assertEqual(_countback().subset_prob(()), 1.0)

    def test_beating_everyone_is_no_easier_than_beating_one(self):
        # P(beat all of S) must be non-increasing as S grows.
        cb = _countback()
        for j in range(len(RIVALS)):
            self.assertLessEqual(cb.subset_prob((0, j)) - 1e-12, cb.subset_prob((j,)))
        self.assertLessEqual(
            cb.subset_prob(tuple(range(len(RIVALS)))) - 1e-12, cb.subset_prob((0,))
        )

    def test_bigger_margin_lead_is_safer(self):
        # I lead Ryan by 55 and Markash by 1, so Ryan must be the safer countback.
        cb = _countback(n_sims=20000)
        ryan = cb.pairwise(0)
        markash = cb.pairwise(2)
        self.assertGreater(ryan, markash)
        self.assertGreater(ryan, 0.80)          # 55 points clear
        self.assertTrue(0.35 < markash < 0.65)  # 1 point clear: near a coin flip

    def test_tipping_the_line_beats_tipping_off_it(self):
        # The median minimises E|M - m|, so the line is the error-minimising tip and
        # must give a better countback than a tip 30 points away from it.
        on_line = T.CountbackModel(ME, RIVALS, [25.0, 25.0], my_margin_tips=[25.0, 25.0],
                                   tau=T.TAU_TIP, n_sims=20000, seed=11)
        off_line = T.CountbackModel(ME, RIVALS, [25.0, 25.0], my_margin_tips=[55.0, 25.0],
                                    tau=T.TAU_TIP, n_sims=20000, seed=11)
        self.assertGreater(on_line.pairwise(0), off_line.pairwise(0))


class TestRivalModel(unittest.TestCase):
    def test_leader_is_modelled_as_always_favourite(self):
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, _flat_fixture())
        leader_group = [g for g in groups if 0 in g.members]
        self.assertEqual(len(leader_group), 1)
        self.assertIsNone(leader_group[0].policy, "leader must tip favourites")

    def test_chasers_are_not_always_favourite(self):
        # Chasers trail the leader by 2 and must deviate to have any chance.
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, _flat_fixture())
        chaser_groups = [g for g in groups if 0 not in g.members]
        self.assertTrue(chaser_groups)
        for g in chaser_groups:
            self.assertIsNotNone(g.policy)

    def test_adriano_separates_from_the_other_145s(self):
        # ADRIANOartini's 646 margin error LOSES the countback to Ryan's 628, so he
        # needs +3 outright where Markash/DeanLFC/Mikefooty need only +2. That must
        # put him in a different policy group.
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, _flat_fixture())
        group_of = {m: gi for gi, g in enumerate(groups) for m in g.members}
        self.assertNotEqual(group_of[5], group_of[2],
                            "ADRIANOartini should not share a policy with Markash")

    def test_every_rival_lands_in_exactly_one_group(self):
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, _flat_fixture())
        members = sorted(m for g in groups for m in g.members)
        self.assertEqual(members, list(range(len(RIVALS))))


class TestJointSolver(unittest.TestCase):
    """The two corollaries the brief asks to be asserted (section 4)."""

    def test_always_favourite_is_a_certain_loss(self):
        # Corollary 1: if I never differentiate, Ryan stays a point ahead and the
        # leaderboard order is frozen. This must be EXACTLY zero.
        p_fav = _flat_fixture()
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, p_fav)
        cb = _countback()
        v = T.evaluate_fixed_policy(ME, RIVALS, p_fav, groups, cb, lambda t, d: "F")
        self.assertAlmostEqual(v, 0.0, places=12)

    def test_optimal_beats_always_favourite(self):
        p_fav = _flat_fixture()
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, p_fav)
        cb = _countback()
        sol = T.solve_joint(ME, RIVALS, p_fav, groups, cb)
        self.assertGreater(sol.p_win, 0.0)
        self.assertLess(sol.p_win, 1.0)

    def test_win_probability_is_a_valid_probability(self):
        p_fav = _flat_fixture(n=6, p=0.6)
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, p_fav)
        cb = _countback()
        sol = T.solve_joint(ME, RIVALS, p_fav, groups, cb)
        for v in (sol.p_win, sol.p_win_favourite, sol.p_win_dog):
            self.assertTrue(0.0 <= v <= 1.0)
        self.assertAlmostEqual(sol.p_win, max(sol.p_win_favourite, sol.p_win_dog), places=12)

    def test_certain_favourites_leave_me_no_path(self):
        # If every favourite is certain, no deviation can ever gain a point, so a
        # one-point deficit is unrecoverable.
        p_fav = [1.0] * 9
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, p_fav)
        cb = _countback()
        sol = T.solve_joint(ME, RIVALS, p_fav, groups, cb)
        self.assertAlmostEqual(sol.p_win, 0.0, places=9)

    def test_more_games_is_never_worse(self):
        cb = _countback()
        prev = -1.0
        for n in (2, 4, 8, 12):
            p_fav = _flat_fixture(n=n, p=0.65)
            groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, p_fav)
            sol = T.solve_joint(ME, RIVALS, p_fav, groups, cb)
            self.assertGreaterEqual(sol.p_win + 1e-9, prev)
            prev = sol.p_win

    def test_solver_is_deterministic(self):
        p_fav = _flat_fixture(n=8, p=0.62)
        results = []
        for _ in range(2):
            cb = T.CountbackModel(ME, RIVALS, [25.0, 25.0], tau=T.TAU_TIP,
                                  n_sims=3000, seed=99)
            groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, p_fav)
            results.append(T.solve_joint(ME, RIVALS, p_fav, groups, cb).p_win)
        self.assertEqual(results[0], results[1])


class TestChaserModel(unittest.TestCase):
    """The pure level-0 chaser satisfices and self-caps, which flatters me. The
    my_assumed_gain knob models a chaser who expects me to be chasing too."""

    # These probe the policy at t=0, so they set reluctance=0 to isolate the
    # my_assumed_gain mechanism. With reluctance on, a chaser facing 18 games of
    # slack simply waits, and t=0 is uniformly "F" whatever the assumed gain.

    def test_satisficing_chasers_stop_deviating_once_they_are_far_enough_ahead(self):
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, _flat_fixture(),
                                      my_assumed_gain=0, reluctance=0.0)
        chaser = [g for g in groups if g.policy is not None][0]
        # There must exist a delta at which they switch to the favourite and stop.
        actions = [chaser.policy[0][d + T.DELTA_CLAMP] for d in range(-2, 6)]
        self.assertIn("F", actions, "a satisficing chaser must stop deviating somewhere")

    def test_assumed_gain_makes_chasers_push_further(self):
        # A chaser who expects me to gain needs a higher delta before they can stop,
        # so their deviation region must extend at least as far.
        def stop_point(gain):
            groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, _flat_fixture(),
                                          my_assumed_gain=gain, reluctance=0.0)
            chaser = [g for g in groups if g.policy is not None][0]
            deltas = [d for d in range(-4, 8)
                      if chaser.policy[0][d + T.DELTA_CLAMP] == "D"]
            return max(deltas) if deltas else -99
        self.assertGreater(stop_point(2), stop_point(0))

    def test_relentless_chasers_do_not_help_me(self):
        # Chasers who account for me must not raise my win probability above the
        # satisficing case, which is the one that flatters me.
        p_fav = _flat_fixture(n=10, p=0.68)
        cb = _countback()
        vals = []
        for gain in (0, 1):
            grp = T.build_rival_groups(RIVALS, [ME] + RIVALS, p_fav, my_assumed_gain=gain)
            vals.append(T.solve_joint(ME, RIVALS, p_fav, grp, cb).p_win)
        self.assertLessEqual(vals[1], vals[0] + 1e-9)


class TestMonteCarloReporting(unittest.TestCase):
    def test_zero_counterexamples_is_not_reported_as_certainty(self):
        # 0 hits in N draws is not proof of 100%.
        self.assertNotIn("100.00", T.pct_mc(1.0, 100_000))
        self.assertTrue(T.pct_mc(1.0, 100_000).startswith(">"))
        self.assertTrue(T.pct_mc(0.0, 100_000).startswith("<"))

    def test_ordinary_probabilities_pass_through(self):
        self.assertEqual(T.pct_mc(0.5, 1000).strip(), "50.00%")


class TestInputValidation(unittest.TestCase):
    def test_missing_leaderboard_file_is_a_clear_error(self):
        with self.assertRaises(T.InputError) as ctx:
            T.load_leaderboard("/nonexistent/leaderboard.csv")
        self.assertIn("--make-template", str(ctx.exception))

    def test_missing_fixtures_file_is_a_clear_error(self):
        with self.assertRaises(T.InputError):
            T.load_fixtures("/nonexistent/fixtures.csv")


def _game(gid, home_odds, away_odds, margin=False, line=25.0):
    return T.Game(gid, "R", "", "Home", "Away", home_odds, away_odds, line, margin)


class TestSeasonSimulation(unittest.TestCase):
    """Ungrouped Monte Carlo: everyone decides again after every result."""

    def _sim(self, me, rivals, games, n_seasons=4000, seed=11):
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        return dict(T.simulate_seasons(me, rivals, games, p_fav,
                                       n_seasons=n_seasons, seed=seed))

    def test_it_is_a_probability_distribution(self):
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(6)]
        table = self._sim(ME, RIVALS, games)
        self.assertAlmostEqual(sum(table.values()), 1.0, places=9)
        for p in table.values():
            self.assertTrue(0.0 <= p <= 1.0)

    def test_every_tipster_appears_exactly_once(self):
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(4)]
        table = self._sim(ME, RIVALS, games)
        self.assertEqual(sorted(table), sorted([ME.name] + [r.name for r in RIVALS]))

    def test_an_uncatchable_leader_wins_every_season(self):
        runaway = T.Tipster("Runaway", ME.points + 9, 900)
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(3)]
        table = self._sim(ME, [runaway], games, n_seasons=500)
        self.assertAlmostEqual(table["Runaway"], 1.0, places=12)

    def test_identical_rivals_get_identical_probabilities(self):
        # The grouped model could not do this: equal points AND equal margin error
        # must give equal chances, up to sampling error.
        twins = [T.Tipster("Twin A", 150, 500), T.Tipster("Twin B", 150, 500)]
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(6)]
        table = self._sim(ME, twins, games, n_seasons=6000)
        self.assertAlmostEqual(table["Twin A"], table["Twin B"], delta=0.03)

    def test_a_rival_behind_a_twin_can_still_win(self):
        # THE POINT OF THE CHANGE. Under grouping, a rival sharing a policy with a
        # higher-pointed twin was exactly 0. Now they chase on their own account.
        ahead = T.Tipster("Ahead", 148, 600)
        behind = T.Tipster("Behind", 147, 600)
        games = [_game("G%d" % i, 1.7, 2.2) for i in range(8)]
        table = self._sim(ME, [ahead, behind], games, n_seasons=6000)
        self.assertGreater(table["Behind"], 0.0,
                           "a lower-pointed rival must have a real chance")

    def test_the_current_leader_tips_favourites(self):
        # One game, all-but-certain favourite, and the leader is only 1 clear. A
        # chaser must take the dog (their only path) while the leader must not.
        leader = T.Tipster("Leader", ME.points + 1, 900)
        games = [_game("G0", 1.01, 30.0)]
        p_fav = [T.favourite_prob(games[0], "odds_ratio")[0]]
        acts = T.simulated_actions(ME, [leader], games, p_fav,
                                   scores=[ME.points, leader.points],
                                   errors=[float(ME.margin_error), 900.0], t=0)
        self.assertEqual(acts[1], "F", "the leader must tip the favourite")
        self.assertEqual(acts[0], "D", "a chaser one back must take its only chance")

    def test_the_leader_rule_follows_the_lead_changing(self):
        # Same two tipsters, but now I am the one in front: the roles must swap.
        # The chaser needs the sharper margin error, otherwise drawing level loses
        # the countback and no tip can save them -- see the test below.
        chaser = T.Tipster("Chaser", ME.points - 1, 100)
        games = [_game("G0", 1.01, 30.0)]
        p_fav = [T.favourite_prob(games[0], "odds_ratio")[0]]
        acts = T.simulated_actions(ME, [chaser], games, p_fav,
                                   scores=[ME.points, chaser.points],
                                   errors=[float(ME.margin_error), 100.0], t=0)
        self.assertEqual(acts[0], "F", "I now lead, so I tip the favourite")
        self.assertEqual(acts[1], "D", "the chaser must deviate")

    def test_a_chaser_who_loses_the_countback_anyway_does_not_bother(self):
        # One game back with the WORSE margin error: drawing level loses the
        # tiebreak, and one game cannot put them two clear. Nothing helps.
        chaser = T.Tipster("Doomed", ME.points - 1, 900)
        games = [_game("G0", 1.01, 30.0)]
        p_fav = [T.favourite_prob(games[0], "odds_ratio")[0]]
        acts = T.simulated_actions(ME, [chaser], games, p_fav,
                                   scores=[ME.points, chaser.points],
                                   errors=[float(ME.margin_error), 900.0], t=0)
        self.assertEqual(acts[1], "F")

    def test_being_level_at_the_top_is_not_leading_if_you_lose_the_countback(self):
        # THE BUG. Level on points with the WORSE margin error is not the lead: a
        # tie loses. Settling for the favourite here settles for defeat, so this
        # tipster must keep deviating.
        blunt = T.Tipster("Blunt", ME.points, ME.margin_error + 100)
        games = [_game("G%d" % i, 1.9, 1.95) for i in range(3)]
        p_fav = [T.favourite_prob(games[0], "odds_ratio")[0]] * 3
        acts = T.simulated_actions(ME, [blunt], games, p_fav,
                                   scores=[ME.points, ME.points],
                                   errors=[float(ME.margin_error),
                                           float(ME.margin_error + 100)], t=0)
        self.assertEqual(acts[1], "D", "level but losing the countback must chase")
        self.assertEqual(acts[0], "F", "level and winning the countback is leading")

    def test_a_blunt_tipster_level_at_the_top_still_wins_sometimes(self):
        # The end-to-end consequence: they must not be locked to exactly zero.
        blunt = T.Tipster("Blunt", ME.points, ME.margin_error + 100)
        games = [_game("G%d" % i, 1.9, 1.95) for i in range(8)]
        table = self._sim(ME, [blunt], games, n_seasons=6000)
        self.assertGreater(table["Blunt"], 0.01)

    def test_the_countback_shapes_how_hard_a_chaser_pushes(self):
        # Two rivals a point back off the same score; the one who would LOSE a tie
        # needs an extra point, so it must deviate at least as often.
        sharp = T.Tipster("Sharp", ME.points - 1, ME.margin_error - 100)
        blunt = T.Tipster("Blunt", ME.points - 1, ME.margin_error + 100)
        games = [_game("G%d" % i, 1.9, 1.95) for i in range(6)]
        p_fav = [T.favourite_prob(games[0], "odds_ratio")[0]] * 6
        acts = T.simulated_actions(ME, [sharp, blunt], games, p_fav,
                                   scores=[ME.points, ME.points - 1, ME.points - 1],
                                   errors=[float(ME.margin_error),
                                           float(ME.margin_error - 100),
                                           float(ME.margin_error + 100)], t=0)
        self.assertEqual(acts[2], "D", "the one who loses the tiebreak must chase")

    def test_the_same_seed_reproduces_the_table(self):
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(5)]
        a = self._sim(ME, RIVALS, games, n_seasons=2000, seed=42)
        b = self._sim(ME, RIVALS, games, n_seasons=2000, seed=42)
        self.assertEqual(a, b)

    def test_different_seeds_agree_within_sampling_error(self):
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(6)]
        a = self._sim(ME, RIVALS, games, n_seasons=8000, seed=1)
        b = self._sim(ME, RIVALS, games, n_seasons=8000, seed=2)
        for name in a:
            self.assertAlmostEqual(a[name], b[name], delta=0.03, msg=name)

    def test_margin_games_break_ties_on_accumulated_error(self):
        # Level on points with one certain favourite left: nobody can separate on
        # tips, so the sharper margin tipper must take it on the countback.
        blunt = T.Tipster("Blunt", ME.points, 5000)
        games = [_game("G0", 1.0001, 5000.0, margin=True)]
        table = self._sim(ME, [blunt], games, n_seasons=500)
        self.assertGreater(table[ME.name], 0.95)


class TestSimulatedRecommendation(unittest.TestCase):
    """The next decision, derived from the simulation itself."""

    def _branches(self, me, rivals, games, n_seasons=4000, seed=13):
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        return T.simulate_branches(me, rivals, games, p_fav,
                                   n_seasons=n_seasons, seed=seed)

    def test_each_branch_is_its_own_distribution(self):
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(5)]
        r = self._branches(ME, RIVALS, games)
        for table in (r.table_favourite, r.table_underdog):
            self.assertAlmostEqual(sum(p for _, p in table), 1.0, places=9)

    def test_my_row_equals_the_headline_for_the_chosen_branch(self):
        # THE ALIGNMENT. Whatever the recommendation is, the table shown must be
        # the table from that branch, and my row in it must be the headline number.
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(5)]
        r = self._branches(ME, RIVALS, games)
        chosen = r.table_underdog if r.action == "D" else r.table_favourite
        self.assertEqual(r.table, chosen)
        self.assertAlmostEqual(dict(r.table)[ME.name], r.p_win, places=12)

    def test_the_recommendation_takes_the_better_branch(self):
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(5)]
        r = self._branches(ME, RIVALS, games)
        self.assertAlmostEqual(r.p_win, max(r.p_win_favourite, r.p_win_underdog),
                               places=12)
        self.assertEqual(r.action, "D" if r.p_win_underdog > r.p_win_favourite else "F")

    def test_each_branch_replays_the_very_same_drawn_seasons(self):
        # The pairing: a branch here must be bit-identical to running the single
        # simulation with that tip forced, on the same seed. Same draws, same
        # replay -- so the two branches differ only by my forced first tip.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(5)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        r = self._branches(ME, RIVALS, games, n_seasons=1500, seed=21)
        for branch, table in (("F", r.table_favourite), ("D", r.table_underdog)):
            solo = T.simulate_seasons(ME, RIVALS, games, p_fav, n_seasons=1500,
                                      seed=21, force_first_me=branch)
            self.assertEqual(table, solo, "branch %s must match the solo run" % branch)

    def test_a_decided_race_has_no_uncertainty_on_the_edge(self):
        # Nobody can catch the leader, so my tip changes nothing in any season:
        # every paired difference is exactly zero, hence so is their spread.
        runaway = T.Tipster("Runaway", ME.points + 40, 1)
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(4)]
        r = self._branches(ME, [runaway], games, n_seasons=500)
        self.assertAlmostEqual(r.stderr_edge, 0.0, places=12)
        self.assertAlmostEqual(r.p_win_favourite, r.p_win_underdog, places=12)

    def test_the_edge_error_is_measured_not_assumed(self):
        # Reported from the realised per-season differences, so it need not match
        # the independence formula -- the branches are not independent.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(6)]
        r = self._branches(ME, RIVALS, games, n_seasons=4000)
        self.assertGreater(r.stderr_edge, 0.0)
        self.assertLess(r.stderr_edge, 0.05)

    def test_taking_a_hopeless_dog_is_never_recommended(self):
        # A near-certain favourite in the next game, and I am already winning:
        # deviating can only cost me, so the favourite must win the comparison.
        games = [_game("G0", 1.0001, 5000.0)] + [_game("G%d" % i, 1.6, 2.4)
                                                 for i in range(1, 4)]
        leader = T.Tipster("Jake Turner", 200, 1, is_me=True)
        r = self._branches(leader, RIVALS, games, n_seasons=1500)
        self.assertEqual(r.action, "F")
        self.assertGreaterEqual(r.p_win_favourite, r.p_win_underdog)

    def test_first_deviation_is_a_distribution(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(6)]
        r = self._branches(ME, RIVALS, games)
        self.assertAlmostEqual(sum(p for _, p in r.first_deviation), 1.0, places=9)
        for idx, p in r.first_deviation:
            self.assertTrue(0.0 <= p <= 1.0)
            self.assertTrue(idx is None or 0 <= idx < len(games))

    def test_it_is_sorted_with_never_reported_separately(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(6)]
        r = self._branches(ME, RIVALS, games)
        indices = [idx for idx, _ in r.first_deviation]
        self.assertEqual(len(indices), len(set(indices)), "one row per outcome")
        probs = [p for idx, p in r.first_deviation if idx is not None]
        self.assertEqual(probs, sorted(probs, reverse=True), "likeliest first")

    def test_deviating_now_makes_this_game_the_answer(self):
        # If the recommendation is to take the dog immediately, the next deviation
        # is this game, in every season.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(6)]
        r = self._branches(ME, RIVALS, games)
        if r.action != "D":
            self.skipTest("this fixture does not recommend deviating now")
        self.assertEqual(r.first_deviation[0], (0, 1.0))

    def test_a_runaway_leader_never_needs_to_deviate(self):
        # So far clear that tipping favourites wins regardless: no deviation.
        leader = T.Tipster("Jake Turner", 400, 1, is_me=True)
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(5)]
        r = self._branches(leader, RIVALS, games, n_seasons=800)
        self.assertAlmostEqual(dict(r.first_deviation).get(None, 0.0), 1.0, places=12)

    def test_a_hopeless_chaser_always_deviates(self):
        # Miles back with the worse countback: every season needs the dog somewhere.
        chaser = T.Tipster("Jake Turner", 100, 9999, is_me=True)
        rivals = [T.Tipster("Runaway", 104, 1)]
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(6)]
        r = self._branches(chaser, rivals, games, n_seasons=800)
        self.assertLess(dict(r.first_deviation).get(None, 0.0), 0.05)

    def test_each_branch_splits_by_the_result_and_reconciles(self):
        # The whole point: P(win) = P(result) * P(win | result) summed over both
        # results. If that identity does not hold the breakdown is decorative.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(5)]
        r = self._branches(ME, RIVALS, games, n_seasons=3000)
        p_hit = r.p_result_favourite
        recomposed_f = (p_hit * r.p_win_favourite_if_hit
                        + (1.0 - p_hit) * r.p_win_favourite_if_miss)
        recomposed_d = ((1.0 - p_hit) * r.p_win_underdog_if_hit
                        + p_hit * r.p_win_underdog_if_miss)
        self.assertAlmostEqual(recomposed_f, r.p_win_favourite, places=9)
        self.assertAlmostEqual(recomposed_d, r.p_win_underdog, places=9)

    def test_the_conditionals_are_probabilities(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(4)]
        r = self._branches(ME, RIVALS, games, n_seasons=1500)
        for v in (r.p_result_favourite, r.p_win_favourite_if_hit,
                  r.p_win_favourite_if_miss, r.p_win_underdog_if_hit,
                  r.p_win_underdog_if_miss):
            self.assertTrue(0.0 <= v <= 1.0)

    def test_getting_your_tip_right_is_never_worse(self):
        # Landing your tip cannot leave you worse off than missing it. Allow a
        # sampling tolerance: when you tip with the field the two conditionals are
        # nearly identical -- everyone moves together -- so the gap is genuinely
        # inside the noise rather than comfortably positive.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(5)]
        r = self._branches(ME, RIVALS, games, n_seasons=4000)
        tol = 3.0 * math.sqrt(0.25 / 4000)
        self.assertGreaterEqual(r.p_win_favourite_if_hit,
                                r.p_win_favourite_if_miss - tol)
        self.assertGreaterEqual(r.p_win_underdog_if_hit,
                                r.p_win_underdog_if_miss - tol)

    def test_the_sampled_result_rate_tracks_the_market(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(4)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        r = self._branches(ME, RIVALS, games, n_seasons=6000)
        self.assertAlmostEqual(r.p_result_favourite, p_fav[0], delta=0.03)

    def test_it_is_reproducible(self):
        games = [_game("G%d" % i, 1.5, 2.6) for i in range(4)]
        a = self._branches(ME, RIVALS, games, n_seasons=1500, seed=8)
        b = self._branches(ME, RIVALS, games, n_seasons=1500, seed=8)
        self.assertEqual(a.table, b.table)
        self.assertEqual(a.p_win_favourite, b.p_win_favourite)
        self.assertEqual(a.p_win_underdog, b.p_win_underdog)


class TestSimulatedContingency(unittest.TestCase):
    """The contingency table must come from the same model as the headline."""

    def _fixture(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(5)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        return games, p_fav

    def test_the_now_cell_is_the_headline_itself(self):
        # THE CONSISTENCY GUARANTEE. Game 0 at delta 0 IS the decision the headline
        # reports, so it must be the identical object, not a re-estimate that could
        # land the other side of a coin flip.
        games, p_fav = self._fixture()
        head = T.simulate_branches(ME, RIVALS, games, p_fav, n_seasons=1200, seed=4)
        cells = T.simulate_contingency(ME, RIVALS, games, p_fav, n_games=2,
                                       deltas=(-1, 0, 1), n_seasons=400, seed=4,
                                       headline=head)
        self.assertIs(cells[(0, 0)], head)

    def test_every_cell_picks_its_own_better_branch(self):
        games, p_fav = self._fixture()
        cells = T.simulate_contingency(ME, RIVALS, games, p_fav, n_games=2,
                                       deltas=(-1, 0, 1), n_seasons=400, seed=4)
        for key, r in cells.items():
            self.assertAlmostEqual(r.p_win, max(r.p_win_favourite, r.p_win_underdog),
                                   places=12, msg=str(key))

    def test_a_bigger_delta_is_never_worse(self):
        # Being further ahead of the all-favourites baseline cannot hurt.
        games, p_fav = self._fixture()
        cells = T.simulate_contingency(ME, RIVALS, games, p_fav, n_games=1,
                                       deltas=(-2, 0, 2), n_seasons=3000, seed=4)
        self.assertLessEqual(cells[(0, -2)].p_win, cells[(0, 0)].p_win + 0.02)
        self.assertLessEqual(cells[(0, 0)].p_win, cells[(0, 2)].p_win + 0.02)

    def test_it_covers_the_requested_grid(self):
        games, p_fav = self._fixture()
        cells = T.simulate_contingency(ME, RIVALS, games, p_fav, n_games=3,
                                       deltas=(-1, 0, 1), n_seasons=300, seed=4)
        self.assertEqual(sorted(cells), sorted((t, d) for t in range(3)
                                               for d in (-1, 0, 1)))

    def test_each_cell_reports_its_own_sampling_error(self):
        games, p_fav = self._fixture()
        cells = T.simulate_contingency(ME, RIVALS, games, p_fav, n_games=1,
                                       deltas=(0,), n_seasons=500, seed=4)
        self.assertGreaterEqual(cells[(0, 0)].stderr_edge, 0.0)


class TestEquilibriumSolver(unittest.TestCase):
    """Backward induction over sampled seasons, rather than a level-0 rollout."""

    def _solve(self, me, rivals, games, n_seasons=600, sweeps=2, seed=3):
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        return T.solve_equilibrium(me, rivals, games, p_fav, n_seasons=n_seasons,
                                   sweeps=sweeps, seed=seed)

    def test_it_learns_a_policy_and_reports_its_own_coverage(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(4)]
        pol = self._solve(ME, RIVALS, games)
        self.assertGreater(pol.n_states, 0)
        self.assertGreaterEqual(pol.min_samples, 1)
        self.assertIsInstance(pol.settled, bool)

    def test_it_is_callable_as_a_decision_rule(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        pol = self._solve(ME, RIVALS, games)
        act = pol(0, T._standing([155, 156], [500.0, 600.0], 0))
        self.assertIn(act, ("F", "D"))

    def test_an_unseen_state_falls_back_rather_than_failing(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        pol = self._solve(ME, RIVALS, games)
        weird = T._standing([100, 400], [1.0, 2.0], 0)
        before = pol.misses
        self.assertIn(pol(0, weird), ("F", "D"))
        self.assertGreater(pol.misses, before)

    def test_a_runaway_leader_is_told_to_hold(self):
        # Nothing can catch them, so every action is a win and the rule must not
        # gratuitously take dogs.
        leader = T.Tipster("Jake Turner", 400, 1, is_me=True)
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        pol = self._solve(leader, RIVALS, games, n_seasons=800)
        self.assertEqual(pol(0, T._standing([400, 150], [1.0, 900.0], 0)), "F")

    def test_a_last_game_deficit_forces_the_dog(self):
        # One game left, a point down, but holding the better margin error so a tie
        # WINS. The leader tips the favourite, so matching them keeps the deficit at
        # one and loses for certain; the dog draws level and takes the countback.
        # Backward induction must find that, since it is the only path.
        games = [_game("G0", 1.6, 2.4)]
        chaser = T.Tipster("Jake Turner", 150, 100, is_me=True)
        rivals = [T.Tipster("Ahead", 151, 900)]
        pol = self._solve(chaser, rivals, games, n_seasons=1500, sweeps=1)
        self.assertEqual(pol(0, T._standing([150, 151], [100.0, 900.0], 0)), "D")

    def test_a_last_game_leader_holds(self):
        games = [_game("G0", 1.6, 2.4)]
        leader = T.Tipster("Jake Turner", 152, 100, is_me=True)
        rivals = [T.Tipster("Behind", 151, 900)]
        pol = self._solve(leader, rivals, games, n_seasons=1500, sweeps=1)
        self.assertEqual(pol(0, T._standing([152, 151], [100.0, 900.0], 0)), "F")

    def test_a_thinly_sampled_state_is_left_to_the_fallback(self):
        # A state seen twice is not knowledge. Writing a coin flip into the table
        # would dress noise up as a decision, so it must be left out and counted.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(4)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        strict = T.solve_equilibrium(ME, RIVALS, games, p_fav, n_seasons=600,
                                     sweeps=1, seed=3, min_visits=10_000)
        self.assertEqual(strict.n_states, 0, "nothing should clear that bar")
        self.assertGreater(strict.thin_states, 0)
        loose = T.solve_equilibrium(ME, RIVALS, games, p_fav, n_seasons=600,
                                    sweeps=1, seed=3, min_visits=1)
        self.assertGreater(loose.n_states, strict.n_states)

    def test_kept_states_all_clear_the_bar(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        pol = T.solve_equilibrium(ME, RIVALS, games, p_fav, n_seasons=1500,
                                  sweeps=1, seed=3, min_visits=20)
        if pol.n_states:
            self.assertGreaterEqual(pol.min_samples, 20)

    def test_the_contingency_grid_queries_absolute_game_indices(self):
        # simulate_contingency slices games[t:], so inside the slice the first game
        # is index 0. A learned policy is keyed by the ABSOLUTE index, so without a
        # shift every cell beyond the first misses its whole policy.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(4)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        seen = []

        def spy(t, standing):
            seen.append(t)
            return "F"

        T.simulate_contingency(ME, RIVALS, games, p_fav, n_games=3, deltas=(0,),
                               n_seasons=40, seed=4, decide=spy)
        self.assertEqual(max(seen), len(games) - 1,
                         "the last game must be reached by its absolute index")

    def test_exploration_draws_from_the_market(self):
        # Exploring a 96% favourite with a coin flip invents states real play never
        # reaches. The explored action must follow the price instead.
        rng = random.Random(5)
        for p in (0.5, 0.62, 0.96):
            favs = sum(1 for _ in range(20000)
                       if T._explore_action(rng.random(), p) == "F")
            self.assertAlmostEqual(favs / 20000.0, p, delta=0.02,
                                   msg="p_fav=%s" % p)

    def test_market_exploration_concentrates_on_lopsided_fixtures(self):
        # Drawing exploratory tips at the market price means a fixture of
        # near-certainties is barely perturbed, so samples concentrate; a fixture of
        # coin flips has genuine variety to cover and must scatter far wider. The
        # comparison is the invariant -- no magic threshold to tune.
        def scatter(home_odds, away_odds):
            games = [_game("G%d" % i, home_odds, away_odds) for i in range(4)]
            p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
            # Concession is switched off here on purpose: it would skip the $1.02
            # fixture outright and there would be no exploration left to measure.
            pol = T.solve_equilibrium(ME, RIVALS, games, p_fav, n_seasons=2000,
                                      sweeps=1, seed=3, min_visits=10,
                                      conceded_odds=1.0)
            self.assertGreater(pol.n_states, 0)
            return pol.thin_states

        heavy = scatter(1.02, 15.0)
        even = scatter(1.95, 1.95)
        self.assertLess(heavy, even / 2.0,
                        "near-certainties should scatter far less than coin flips")

    def test_the_csv_is_written_without_crashing(self):
        # The countback rows reach for a model the writer never pulled out of the
        # result dict, so every single run died after printing the whole report.
        import os
        import tempfile
        me = T.Tipster("Jake Turner", 155, 577, is_me=True)
        rivals = [T.Tipster("Ryan Board", 156, 628)]
        games = [_game("G0", 1.6, 2.4, margin=True), _game("G1", 1.8, 2.0)]
        with tempfile.TemporaryDirectory() as tmp:
            paths = T.SetPaths(name="current",
                               leaderboard=os.path.join(tmp, "leaderboard.csv"),
                               fixtures=os.path.join(tmp, "fixtures.csv"),
                               output_dir=tmp)
            result = T.report(me, rivals, games, "odds_ratio", False, 2000,
                              1, T.TAU_TIP, paths, n_seasons=500,
                              contingency_seasons=200)
            out = T.write_csv_outputs(me, rivals, games, result, "odds_ratio", paths)
            with open(out) as fh:
                body = fh.read()
        self.assertIn("countback", body)

    def test_both_arms_are_sampled_for_every_visited_state(self):
        # Playing both arms over the SAME drawn season is what guarantees this: a
        # state can no longer be reached with one arm sampled and the other missing,
        # which used to throw the state away. So no state should be discarded for a
        # missing arm -- only ever for being under the sample bar.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        pol = T.solve_equilibrium(ME, RIVALS, games, p_fav, n_seasons=400,
                                  sweeps=1, seed=3, min_visits=1)
        self.assertGreater(pol.n_states, 0)
        self.assertEqual(pol.thin_states, 0,
                         "with paired arms nothing can be missing an arm at min 1")

    def test_every_tipster_contributes_a_sample(self):
        # One season used to yield a single data point from one random focus. Scoring
        # the whole field multiplies the yield, so the same seasons must learn
        # strictly more states than a single-focus run could.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        pol = T.solve_equilibrium(ME, RIVALS, games, p_fav, n_seasons=500,
                                  sweeps=1, seed=3, min_visits=25)
        # Each season yields 2 arms x the whole field; a state clearing 25 per arm
        # off 500 seasons is only reachable at that yield.
        self.assertGreater(pol.n_states, 0)
        self.assertGreaterEqual(pol.min_samples, 25)

    def test_short_priced_games_are_conceded_not_solved(self):
        # If everyone tips it, every score moves together and no gap changes, so
        # there is nothing to decide. The rule must say so without consulting the
        # table, and without being counted as a learned hit or a fallback.
        games = [_game("G0", 1.06, 11.0), _game("G1", 1.6, 2.4)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        pol = T.solve_equilibrium(ME, RIVALS, games, p_fav, n_seasons=400,
                                  sweeps=1, seed=3)
        self.assertEqual(pol.conceded, frozenset([0]))
        self.assertFalse(any(t == 0 for t, _ in pol.table),
                         "a conceded game should never be solved")
        hits, misses = pol.hits, pol.misses
        self.assertEqual(pol(0, T._standing([155, 156], [500.0, 600.0], 0)), "F")
        self.assertEqual((pol.hits, pol.misses), (hits, misses))
        self.assertEqual(pol.conceded_calls, 1)

    def test_concession_reads_the_price_not_the_favourite_side(self):
        # The short price can sit on either side of the fixture.
        games = [_game("G0", 11.0, 1.06), _game("G1", 1.30, 3.6)]
        self.assertEqual(T.conceded_games(games, T.CONCEDED_ODDS), frozenset([0]))
        self.assertEqual(T.conceded_games(games, 1.0), frozenset(),
                         "a threshold at evens must disable the assumption")

    def test_the_coarse_key_buckets_only_the_table_lookup(self):
        # Clamping is applied to the learned table's key. The fallback needs the true
        # multiset to compute terminal values from, so it must receive that unchanged.
        exact = T._standing([155, 161], [500.0, 600.0], 0)
        self.assertEqual(T._coarse(exact, None), exact)
        self.assertEqual(T._coarse(exact, 2), ((2, 1),))
        seen = []

        def fallback(t, standing):
            seen.append(standing)
            return "F"

        pol = T.EquilibriumPolicy({}, fallback, 1, 1, 0, 0.0, gap_clamp=2)
        pol(0, exact)
        self.assertEqual(seen, [exact], "the fallback must see the true gaps")

    def test_convergence_is_reported_as_a_rate_not_a_boolean(self):
        # A boolean "nothing moved" can only ever read False across hundreds of
        # sampled states, so it carries no information. The rate does.
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        pol = T.solve_equilibrium(ME, RIVALS, games, p_fav, n_seasons=400,
                                  sweeps=2, seed=3)
        self.assertGreaterEqual(pol.flip_rate, 0.0)
        self.assertLessEqual(pol.flip_rate, 1.0)
        self.assertEqual(pol.settled, pol.flip_rate <= T.SETTLED_TOL)

    def test_the_field_board_covers_everyone_for_the_next_game(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        pol = self._solve(ME, RIVALS, games)
        board = T.field_tips(ME, RIVALS, games, 0, pol)
        self.assertEqual([row[0] for row in board],
                         [ME.name] + [r.name for r in RIVALS])
        for name, action, team, gap, _q in board:
            self.assertIn(action, ("F", "D"))
            self.assertIn(team, (games[0].home, games[0].away))
            self.assertIsInstance(gap, int)

    def test_the_field_board_names_the_team_the_action_means(self):
        # Home is the favourite here, so "F" must read as the home side.
        games = [_game("G0", 1.4, 3.0)]
        pol = self._solve(ME, RIVALS, games, n_seasons=800)
        for _, action, team, _, _q in T.field_tips(ME, RIVALS, games, 0, pol):
            self.assertEqual(team, games[0].home if action == "F" else games[0].away)

    def test_the_field_board_reports_the_gap_to_the_lead(self):
        leader = T.Tipster("Top", ME.points + 3, 100)
        games = [_game("G0", 1.6, 2.4)]
        pol = self._solve(ME, [leader], games, n_seasons=800)
        board = dict((row[0], row[3]) for row in T.field_tips(ME, [leader], games, 0, pol))
        self.assertEqual(board["Top"], 0, "the leader is level with the lead")
        self.assertEqual(board[ME.name], -3)

    def test_the_same_seed_learns_the_same_policy(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(3)]
        a = self._solve(ME, RIVALS, games)
        b = self._solve(ME, RIVALS, games)
        self.assertEqual(a.table, b.table)

    def test_the_learned_policy_drives_the_simulation(self):
        games = [_game("G%d" % i, 1.6, 2.4) for i in range(4)]
        p_fav = [T.favourite_prob(g, "odds_ratio")[0] for g in games]
        pol = self._solve(ME, RIVALS, games)
        r = T.simulate_branches(ME, RIVALS, games, p_fav, n_seasons=800, seed=6,
                                decide=pol)
        self.assertAlmostEqual(sum(p for _, p in r.table), 1.0, places=9)
        self.assertIn(r.action, ("F", "D"))


class TestDeviationReluctance(unittest.TestCase):
    """Rivals should take dogs readily in close games and rarely in lopsided ones."""

    # A chaser needing ground, over a fixture mixing close and lopsided games.
    FIXTURE = [0.55, 0.70, 0.85, 0.96, 0.60, 0.78]

    def _chaser_policy(self, reluctance):
        target = lambda d: 1.0 if d >= 2 else 0.0
        _, policy = T.solve_level0(self.FIXTURE, target, reluctance=reluctance)
        return [policy[t][0 + T.DELTA_CLAMP] for t in range(len(self.FIXTURE))]

    def test_zero_reluctance_reproduces_todays_policy(self):
        _, baseline = T.solve_level0(self.FIXTURE, lambda d: 1.0 if d >= 2 else 0.0)
        _, explicit = T.solve_level0(self.FIXTURE, lambda d: 1.0 if d >= 2 else 0.0,
                                     reluctance=0.0)
        self.assertEqual(baseline, explicit)

    def test_reluctance_never_increases_deviation(self):
        counts = [self._chaser_policy(k).count("D") for k in (0.0, 0.1, 0.3, 0.8)]
        for earlier, later in zip(counts, counts[1:]):
            self.assertLessEqual(later, earlier, "more reluctance must not deviate more")
        self.assertLess(counts[-1], counts[0], "high reluctance must deviate strictly less")

    def test_heavy_favourites_are_dropped_before_close_games(self):
        # Raise reluctance until deviation first disappears somewhere. The game it
        # abandons first must not be a closer game than one it still deviates in.
        for k in (0.05, 0.1, 0.2, 0.4, 0.8):
            actions = self._chaser_policy(k)
            deviating = [self.FIXTURE[t] for t, a in enumerate(actions) if a == "D"]
            skipped = [self.FIXTURE[t] for t, a in enumerate(actions) if a == "F"]
            if deviating and skipped:
                self.assertLessEqual(
                    min(deviating), max(skipped) + 1e-12,
                    "at k=%s it deviates at %.2f but skips %.2f" % (
                        k, min(deviating), max(skipped)),
                )

    def test_reluctance_never_prefers_a_certain_loss_to_a_live_chance(self):
        # One game, a 98% favourite, and the dog is the ONLY path to first. The
        # penalty (0.048) exceeds the dog's value (0.018), so an unguarded additive
        # rule would take the certain loss. It must not.
        _, policy = T.solve_level0([0.982], lambda d: 1.0 if d >= 1 else 0.0,
                                   reluctance=T.RELUCTANCE)
        self.assertEqual(policy[0][T.DELTA_CLAMP], "D")

    def test_reluctance_still_bites_when_there_is_something_to_protect(self):
        # Same heavy favourite, but now standing pat already wins outright, so the
        # tipster has a live alternative and reluctance should hold them back.
        _, policy = T.solve_level0([0.982], lambda d: 1.0 if d >= 0 else 0.0,
                                   reluctance=T.RELUCTANCE)
        self.assertEqual(policy[0][T.DELTA_CLAMP], "F")

    def test_a_coin_flip_game_is_never_penalised(self):
        # penalty = k * max(0, p - 0.5), so p = 0.5 costs nothing at any k.
        even = [0.5, 0.5, 0.5]
        target = lambda d: 1.0 if d >= 1 else 0.0
        _, base = T.solve_level0(even, target, reluctance=0.0)
        _, harsh = T.solve_level0(even, target, reluctance=5.0)
        self.assertEqual(base, harsh)

    def test_values_stay_real_probabilities(self):
        # The penalty selects the action; it must not leak into the stored value.
        values, _ = T.solve_level0(self.FIXTURE, lambda d: 1.0 if d >= 2 else 0.0,
                                   reluctance=0.5)
        for row in values:
            for v in row:
                self.assertTrue(0.0 <= v <= 1.0, "value %r is not a probability" % v)

    def test_reluctance_reaches_the_rival_groups(self):
        p_fav = [0.55, 0.62, 0.88, 0.95, 0.60, 0.72, 0.66, 0.90]

        def deviations(reluctance):
            groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, p_fav,
                                          reluctance=reluctance)
            chasers = [g for g in groups if g.policy is not None]
            return sum(
                sum(1 for t in range(len(p_fav))
                    if g.policy[t][T.DELTA_CLAMP] == "D")
                for g in chasers
            )

        self.assertLess(deviations(0.8), deviations(0.0))

    def test_default_reluctance_is_documented_and_nonzero(self):
        self.assertGreater(T.RELUCTANCE, 0.0)
        self.assertLess(T.RELUCTANCE, 1.0)


class TestStandingsNarrative(unittest.TestCase):
    """The WHY text must derive the standings, not assert them."""

    def test_trailing_is_reported_as_trailing(self):
        text = T.standings_summary(ME, RIVALS)
        self.assertIn("trail", text)
        self.assertIn("Ryan Board", text)

    def test_leading_is_not_reported_as_a_negative_deficit(self):
        leader = T.Tipster("Jake Turner", 200, 573, is_me=True)
        text = T.standings_summary(leader, RIVALS)
        self.assertNotIn("-", text)
        self.assertIn("lead", text)

    def test_a_tie_at_the_top_is_reported_as_level(self):
        tied = T.Tipster("Jake Turner", 147, 573, is_me=True)
        text = T.standings_summary(tied, RIVALS)
        self.assertIn("level", text)

    def test_the_margin_error_claim_is_checked_not_asserted(self):
        # Give a rival a better margin error than mine; the text must not claim
        # that a tie for first is a win for me against every rival.
        rivals = [T.Tipster("Sharp", 147, 100)]
        text = T.standings_summary(ME, rivals)
        self.assertNotIn("lowest margin error", text)
        self.assertIn("Sharp", text)

    def test_the_margin_error_claim_is_made_when_it_holds(self):
        text = T.standings_summary(ME, RIVALS)
        self.assertIn("lowest margin error", text)


class TestInputSets(unittest.TestCase):
    def test_resolve_builds_input_and_output_paths(self):
        p = T.resolve_set("scenario", input_dir="/in", output_dir="/out")
        self.assertEqual(p.name, "scenario")
        self.assertEqual(p.leaderboard, "/in/scenario/leaderboard.csv")
        self.assertEqual(p.fixtures, "/in/scenario/fixtures.csv")
        self.assertEqual(p.output_dir, "/out/scenario")

    def test_resolve_defaults_to_the_project_directories(self):
        p = T.resolve_set(T.DEFAULT_SET)
        self.assertEqual(p.leaderboard, T.LEADERBOARD_CSV)
        self.assertEqual(p.fixtures, T.FIXTURES_CSV)

    def test_default_set_is_current(self):
        self.assertEqual(T.DEFAULT_SET, "current")
        self.assertTrue(T.LEADERBOARD_CSV.endswith("inputs/current/leaderboard.csv"))

    def test_set_name_may_not_escape_the_inputs_directory(self):
        for bad in ("../secrets", "a/b", "", "."):
            with self.assertRaises(T.InputError, msg=bad):
                T.resolve_set(bad)

    def test_template_writes_into_the_named_set(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            paths = T.resolve_set("sandbox", input_dir=tmp, output_dir=tmp)
            T.write_template(paths)
            me, rivals = T.load_leaderboard(paths.leaderboard)
            self.assertTrue(me.is_me)
            self.assertEqual(len(rivals), 6)
            # The fixture template is deliberately unfilled, so loading must fail
            # with the message telling you to fill it in.
            with self.assertRaises(T.InputError) as ctx:
                T.load_fixtures(paths.fixtures)
            self.assertIn("no completed rows", str(ctx.exception))

    def _seeded_pair(self, tmp):
        """A populated source set and an empty dest set inside tmp."""
        src = T.resolve_set("current", input_dir=tmp, output_dir=tmp)
        dst = T.resolve_set("scenario", input_dir=tmp, output_dir=tmp)
        T.write_template(src)
        return src, dst

    def test_copy_set_clones_both_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = self._seeded_pair(tmp)
            written = T.copy_set(src, dst)
            self.assertEqual(written, [dst.leaderboard, dst.fixtures])
            with open(src.leaderboard) as a, open(dst.leaderboard) as b:
                self.assertEqual(a.read(), b.read())

    def test_copy_set_refuses_a_missing_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = T.resolve_set("current", input_dir=tmp, output_dir=tmp)
            dst = T.resolve_set("scenario", input_dir=tmp, output_dir=tmp)
            with self.assertRaises(T.InputError) as ctx:
                T.copy_set(src, dst)
            self.assertIn("current", str(ctx.exception))

    def test_copy_set_refuses_to_copy_onto_itself(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src, _ = self._seeded_pair(tmp)
            with self.assertRaises(T.InputError):
                T.copy_set(src, src)

    def test_copy_set_will_not_clobber_without_confirmation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = self._seeded_pair(tmp)
            T.copy_set(src, dst)
            with open(dst.fixtures, "w") as fh:
                fh.write("hand-edited\n")
            written = T.copy_set(src, dst, confirm=lambda prompt: "n")
            self.assertEqual(written, [])
            with open(dst.fixtures) as fh:
                self.assertEqual(fh.read(), "hand-edited\n", "declining must not overwrite")

    def test_copy_set_clobbers_when_confirmed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = self._seeded_pair(tmp)
            T.copy_set(src, dst)
            with open(dst.fixtures, "w") as fh:
                fh.write("hand-edited\n")
            self.assertTrue(T.copy_set(src, dst, confirm=lambda prompt: "y"))
            with open(dst.fixtures) as fh:
                self.assertNotEqual(fh.read(), "hand-edited\n")

    def test_copy_set_force_skips_the_prompt(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = self._seeded_pair(tmp)
            T.copy_set(src, dst)

            def explode(prompt):
                raise AssertionError("--force must not prompt")

            self.assertTrue(T.copy_set(src, dst, force=True, confirm=explode))

    def test_explicit_paths_override_the_set(self):
        paths, _ = T.effective_paths("scenario", "/tmp/lb.csv", "/tmp/fx.csv")
        self.assertEqual(paths.leaderboard, "/tmp/lb.csv")
        self.assertEqual(paths.fixtures, "/tmp/fx.csv")
        self.assertTrue(paths.output_dir.endswith("scenario"),
                        "output still belongs to the named set")

    def test_no_override_leaves_the_set_paths_alone(self):
        paths, warnings = T.effective_paths("scenario", None, None)
        self.assertEqual(paths, T.resolve_set("scenario"))
        self.assertEqual(warnings, [])

    def test_overriding_exactly_one_path_warns(self):
        _, warnings = T.effective_paths("current", "/tmp/lb.csv", None)
        self.assertEqual(len(warnings), 1)
        self.assertIn("/tmp/lb.csv", warnings[0])
        self.assertIn("fixtures", warnings[0])

    def test_overriding_both_paths_does_not_warn(self):
        _, warnings = T.effective_paths("current", "/tmp/lb.csv", "/tmp/fx.csv")
        self.assertEqual(warnings, [])

    def test_scenario_banner_fires_for_non_default_sets(self):
        self.assertIn("NOT REALITY", T.set_banner(T.resolve_set("scenario")))
        self.assertIn("scenario", T.set_banner(T.resolve_set("scenario")))

    def test_no_scenario_banner_for_the_current_set(self):
        self.assertNotIn("NOT REALITY", T.set_banner(T.resolve_set(T.DEFAULT_SET)))




class TestSoftLevel0(unittest.TestCase):
    """Probabilistic deviation: a logistic on the edge, with soft value iteration."""

    FIXTURE = [0.585, 0.655, 0.66, 0.68, 0.69, 0.76, 0.853, 0.96]

    def target(self, need):
        return lambda d: 1.0 if d >= need else 0.0

    def test_zero_temperature_reproduces_the_hard_policy(self):
        # The whole backwards-compatibility guarantee in one assertion.
        for reluctance in (0.0, 0.10, 0.5):
            values, policy = T.solve_level0(self.FIXTURE, self.target(2),
                                            reluctance=reluctance)
            soft_v, dog_p = T.solve_level0_soft(self.FIXTURE, self.target(2),
                                                reluctance=reluctance,
                                                temperature=0.0)
            self.assertEqual(values, soft_v)
            for t, row in enumerate(policy):
                for i, act in enumerate(row):
                    self.assertIn(dog_p[t][i], (0.0, 1.0))
                    self.assertEqual(act, "D" if dog_p[t][i] == 1.0 else "F")

    def test_probabilities_are_valid(self):
        _, dog_p = T.solve_level0_soft(self.FIXTURE, self.target(2),
                                       reluctance=0.10, temperature=0.05)
        for row in dog_p:
            for q in row:
                self.assertGreaterEqual(q, 0.0)
                self.assertLessEqual(q, 1.0)

    def test_values_remain_probabilities(self):
        for temperature in (0.0, 0.02, 0.05, 0.15, 1.0):
            values, _ = T.solve_level0_soft(self.FIXTURE, self.target(2),
                                            reluctance=0.10,
                                            temperature=temperature)
            for row in values:
                for v in row:
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 1.0)

    def test_a_zero_edge_is_a_coin_flip(self):
        # One game, and the target is met either way, so taking the dog neither
        # gains nor costs: v_dog == v_fav, the game is live, so q == 0.5.
        # reluctance=0 keeps the penalty out of it.
        _, dog_p = T.solve_level0_soft([0.5], lambda d: 1.0, clamp=3,
                                       reluctance=0.0, temperature=0.05)
        self.assertAlmostEqual(dog_p[0][3], 0.5, places=9)

    def test_a_dead_state_does_not_coin_flip(self):
        # Cannot win from anywhere: v_fav == v_dog == 0. The logistic would say
        # 50%, which would have eliminated rivals flipping coins every game.
        _, dog_p = T.solve_level0_soft(self.FIXTURE, lambda d: 0.0,
                                       reluctance=0.10, temperature=0.05)
        for row in dog_p:
            for q in row:
                self.assertEqual(q, 0.0)

    def test_probability_is_monotone_in_the_edge(self):
        # Raising the bar the tipster must clear can only make the dog more
        # attractive at delta 0, never less.
        _, easy = T.solve_level0_soft(self.FIXTURE, self.target(0),
                                      reluctance=0.0, temperature=0.05)
        _, hard = T.solve_level0_soft(self.FIXTURE, self.target(3),
                                      reluctance=0.0, temperature=0.05)
        self.assertLess(easy[0][T.DELTA_CLAMP], hard[0][T.DELTA_CLAMP])

    def test_higher_temperature_pulls_towards_a_coin_flip(self):
        # A confident state at low temperature must get less confident as the
        # temperature rises.
        sharp = T.solve_level0_soft(self.FIXTURE, self.target(2),
                                    reluctance=0.10, temperature=0.01)[1]
        blunt = T.solve_level0_soft(self.FIXTURE, self.target(2),
                                    reluctance=0.10, temperature=0.30)[1]
        i = T.DELTA_CLAMP
        for t in range(len(self.FIXTURE)):
            if sharp[t][i] in (0.0, 1.0):
                continue
            self.assertLess(abs(blunt[t][i] - 0.5), abs(sharp[t][i] - 0.5) + 1e-12)

    def test_noise_costs_the_tipster_value(self):
        # Acting noisily cannot be worth more than acting optimally.
        opt, _ = T.solve_level0_soft(self.FIXTURE, self.target(2),
                                     reluctance=0.0, temperature=0.0)
        noisy, _ = T.solve_level0_soft(self.FIXTURE, self.target(2),
                                       reluctance=0.0, temperature=0.05)
        self.assertLessEqual(noisy[0][T.DELTA_CLAMP],
                             opt[0][T.DELTA_CLAMP] + 1e-12)

    def test_default_constant(self):
        self.assertEqual(T.RIVAL_NOISE, 0.05)


class TestNoisyDecisions(unittest.TestCase):
    """The simulation's decision rule, once rivals stop being certain."""

    GAMES = [
        T.Game("G1", "R", "Thu", "A", "B", 1.40, 2.96, 16.5, True, None),
        T.Game("G2", "R", "Fri", "C", "D", 2.44, 1.56, 10.5, False, None),
        T.Game("G3", "R", "Sat", "E", "F", 2.22, 1.67, 4.5, False, None),
    ]
    P_FAV = [0.689, 0.616, 0.574]

    def test_cache_returns_zero_or_one_at_zero_temperature(self):
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP)
        standing = ((1, -1), (0, 1))
        for t in range(3):
            self.assertIn(decide(t, standing), (0.0, 1.0))

    def test_cache_returns_a_strict_probability_under_noise(self):
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        q = decide(0, ((1, -1), (0, 1)))
        self.assertGreater(q, 0.0)
        self.assertLess(q, 1.0)

    def test_actions_are_deterministic_when_no_draws_are_supplied(self):
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        scores, errors = [155, 156, 154], [577.0, 628.0, 583.0]
        first = T._actions(scores, errors, 0, decide)
        for _ in range(5):
            self.assertEqual(T._actions(scores, errors, 0, decide), first)

    def test_draws_select_the_action(self):
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        scores, errors = [155, 156, 154], [577.0, 628.0, 583.0]
        all_low = T._actions(scores, errors, 0, decide, draws=[0.0, 0.0, 0.0])
        all_high = T._actions(scores, errors, 0, decide, draws=[0.0, 1.0, 1.0])
        # Index 0 is me and never draws, so only the rivals may differ.
        self.assertEqual(all_low[0], all_high[0])
        self.assertNotEqual(all_low[1:], all_high[1:])

    def test_i_never_draw(self):
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        scores, errors = [155, 156, 154], [577.0, 628.0, 583.0]
        mine = T._actions(scores, errors, 0, decide, draws=[0.0, 0.5, 0.5])[0]
        for u in (0.0, 0.25, 0.5, 0.75, 1.0):
            got = T._actions(scores, errors, 0, decide, draws=[u, 0.5, 0.5])[0]
            self.assertEqual(got, mine)

    def test_decide_me_overrides_index_zero_only(self):
        rivals_rule = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                        temperature=0.05)
        scores, errors = [155, 156, 154], [577.0, 628.0, 583.0]
        always_dog = lambda t, standing: 1.0
        acts = T._actions(scores, errors, 0, rivals_rule,
                          draws=[0.0, 0.0, 0.0], decide_me=always_dog)
        self.assertEqual(acts[0], "D")

    def test_equilibrium_policy_keeps_one_contract_on_every_path(self):
        # It answers with an action whether the state was learned or fell back,
        # even when the fallback underneath it is a noisy probability.
        noisy = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                  temperature=0.05)
        policy = T.EquilibriumPolicy({}, noisy, 100, 1, 0, 0.0)
        self.assertIn(policy(0, ((1, -1),)), ("F", "D"))
        table = {(0, ((1, -1),)): "D"}
        learned = T.EquilibriumPolicy(table, noisy, 100, 1, 0, 0.0)
        self.assertEqual(learned(0, ((1, -1),)), "D")

    def test_as_dog_prob_accepts_both_forms(self):
        self.assertEqual(T._as_dog_prob("D"), 1.0)
        self.assertEqual(T._as_dog_prob("F"), 0.0)
        self.assertEqual(T._as_dog_prob(0.37), 0.37)


class TestNoisePreDraw(unittest.TestCase):
    """Rival noise is luck, so it is drawn once per season and shared by branches."""

    GAMES = [
        T.Game("G1", "R", "Thu", "A", "B", 1.40, 2.96, 16.5, True, None),
        T.Game("G2", "R", "Fri", "C", "D", 2.44, 1.56, 10.5, False, None),
    ]
    P_FAV = [0.689, 0.616]
    ME = T.Tipster("Me", 155, 577, is_me=True)
    RIVALS = [T.Tipster("Leader", 156, 628), T.Tipster("Chaser", 154, 583)]

    def test_draw_season_returns_noise_shaped_per_game_per_tipster(self):
        rng = random.Random(1)
        results, margins, noise = T._draw_season(
            self.GAMES, self.P_FAV, 2, rng.random, rng.gauss, 10.0, 37.0,
            draw_noise=True)
        self.assertEqual(len(noise), len(self.GAMES))
        for row in noise:
            self.assertEqual(len(row), 3)          # me + two rivals
            for u in row:
                self.assertGreaterEqual(u, 0.0)
                self.assertLess(u, 1.0)

    def test_replaying_one_season_is_reproducible(self):
        rng = random.Random(7)
        results, margins, noise = T._draw_season(
            self.GAMES, self.P_FAV, 2, rng.random, rng.gauss, 10.0, 37.0,
            draw_noise=True)
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        scores, errors = [155, 156, 154], [577.0, 628.0, 583.0]
        first = T._play_season(self.GAMES, scores, errors, results, margins,
                               decide, "F", noise=noise)
        for _ in range(5):
            self.assertEqual(
                T._play_season(self.GAMES, scores, errors, results, margins,
                               decide, "F", noise=noise),
                first)

    def test_both_branches_see_the_same_noise(self):
        # The pairing is the whole point: the only thing that may differ between
        # the branches is my forced first tip, never the rivals' luck.
        rng = random.Random(7)
        _, _, noise = T._draw_season(self.GAMES, self.P_FAV, 2,
                                     rng.random, rng.gauss, 10.0, 37.0,
                                     draw_noise=True)
        seen = {}
        for branch in ("F", "D"):
            drawn = []

            class Spy:
                def __init__(self, inner):
                    self.inner = inner

                def __call__(self, t, standing):
                    drawn.append(t)
                    return self.inner(t, standing)

            decide = Spy(T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                           temperature=0.05))
            T._play_season(self.GAMES, [155, 156, 154], [577.0, 628.0, 583.0],
                           [True, False], {}, decide, branch, noise=noise)
            seen[branch] = list(noise)
        self.assertEqual(seen["F"], seen["D"])

    def test_simulate_branches_is_deterministic_under_noise(self):
        kwargs = dict(n_seasons=400, seed=99, temperature=0.05)
        a = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                **kwargs)
        b = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                **kwargs)
        self.assertEqual(a.p_win_favourite, b.p_win_favourite)
        self.assertEqual(a.p_win_underdog, b.p_win_underdog)

    def test_zero_temperature_matches_the_default(self):
        quiet = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                    n_seasons=800, seed=5, temperature=0.0)
        default = T.simulate_branches(self.ME, self.RIVALS, self.GAMES,
                                      self.P_FAV, n_seasons=800, seed=5)
        self.assertEqual(quiet.p_win_favourite, default.p_win_favourite)
        self.assertEqual(quiet.p_win_underdog, default.p_win_underdog)

    def test_noise_off_does_not_touch_the_rng_stream(self):
        # Drawing noise unconditionally would shift every later draw, so a run
        # with noise off would stop reproducing pre-feature numbers.
        rng_a = random.Random(4)
        quiet = T._draw_season(self.GAMES, self.P_FAV, 2, rng_a.random,
                               rng_a.gauss, 10.0, 37.0)
        rng_b = random.Random(4)
        loud = T._draw_season(self.GAMES, self.P_FAV, 2, rng_b.random,
                              rng_b.gauss, 10.0, 37.0, draw_noise=True)
        self.assertIsNone(quiet[2])
        self.assertIsNotNone(loud[2])
        self.assertEqual(quiet[0], loud[0])          # same results
        self.assertNotEqual(quiet[1], loud[1])       # margins necessarily shift

        # The property that matters: noise off consumes exactly the draws it
        # always did, so nothing downstream in the stream moves.
        def counted(seed, draw_noise):
            rng = random.Random(seed)
            calls = []

            def rand():
                calls.append(1)
                return rng.random()

            T._draw_season(self.GAMES, self.P_FAV, 2, rand, rng.gauss,
                           10.0, 37.0, draw_noise=draw_noise)
            return len(calls)

        n_games, n_tipsters = len(self.GAMES), 3
        self.assertEqual(counted(4, False), n_games)
        self.assertEqual(counted(4, True), n_games + n_games * n_tipsters)

    def test_the_bloc_actually_breaks(self):
        # Two rivals in identical positions must sometimes tip differently within
        # one season. Under the old hard argmax they never could.
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        rng = random.Random(3)
        split = False
        for _ in range(200):
            _, _, noise = T._draw_season(self.GAMES, self.P_FAV, 2,
                                         rng.random, rng.gauss, 10.0, 37.0,
                                         draw_noise=True)
            acts = T._actions([155, 154, 154], [577.0, 590.0, 590.0], 0,
                              decide, draws=noise[0])
            if acts[1] != acts[2]:
                split = True
                break
        self.assertTrue(split, "identically placed rivals never disagreed")

    def test_noise_changes_the_field(self):
        quiet = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                    n_seasons=3000, seed=11, temperature=0.0)
        noisy = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                    n_seasons=3000, seed=11, temperature=0.15)
        self.assertNotEqual(quiet.table_favourite, noisy.table_favourite)


class TestNoiseSurface(unittest.TestCase):
    """The flag, the reported probability, and the assumptions block."""

    GAMES = [
        T.Game("G1", "R", "Thu", "A", "B", 1.40, 2.96, 16.5, True, None),
        T.Game("G2", "R", "Fri", "C", "D", 2.44, 1.56, 10.5, False, None),
    ]
    P_FAV = [0.689, 0.616]
    ME = T.Tipster("Me", 155, 577, is_me=True)
    RIVALS = [T.Tipster("Leader", 156, 628), T.Tipster("Chaser", 154, 583)]

    def test_field_tips_reports_the_probability(self):
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        rows = T.field_tips(self.ME, self.RIVALS, self.GAMES, 0, decide)
        for name, act, team, gap, q in rows:
            self.assertIn(act, ("F", "D"))
            self.assertGreaterEqual(q, 0.0)
            self.assertLessEqual(q, 1.0)
            self.assertEqual(act, "D" if q > 0.5 else "F")

    def test_field_tips_is_certain_at_zero_temperature(self):
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP)
        for _, _, _, _, q in T.field_tips(self.ME, self.RIVALS, self.GAMES, 0,
                                          decide):
            self.assertIn(q, (0.0, 1.0))

    def test_cli_exposes_rival_noise(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                T.main(["--help"])
        self.assertIn("--rival-noise", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
