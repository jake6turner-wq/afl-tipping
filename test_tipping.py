#!/usr/bin/env python3
"""Tests for the AFL tipping engine. Pure stdlib unittest, no third-party deps.

Run:  python3 test_tipping.py
"""

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

    def test_satisficing_chasers_stop_deviating_once_they_are_far_enough_ahead(self):
        groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, _flat_fixture(),
                                      my_assumed_gain=0)
        chaser = [g for g in groups if g.policy is not None][0]
        # There must exist a delta at which they switch to the favourite and stop.
        actions = [chaser.policy[0][d + T.DELTA_CLAMP] for d in range(-2, 6)]
        self.assertIn("F", actions, "a satisficing chaser must stop deviating somewhere")

    def test_assumed_gain_makes_chasers_push_further(self):
        # A chaser who expects me to gain needs a higher delta before they can stop,
        # so their deviation region must extend at least as far.
        def stop_point(gain):
            groups = T.build_rival_groups(RIVALS, [ME] + RIVALS, _flat_fixture(),
                                          my_assumed_gain=gain)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
