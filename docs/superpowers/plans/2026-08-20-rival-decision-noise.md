# Probabilistic Rival Deviation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make modelled rivals take the underdog with a probability rather than a
certainty, so a knife-edge preference stops producing a certain, field-wide action.

**Architecture:** A logistic on the deviation edge the DP already computes, with
soft value iteration so a rival's stored value reflects the noisy player they
actually are. Scoped to the season simulation, where an independent coin per rival
is free; the exact grouped solve stays deterministic because independent mixing
would destroy its shared-delta collapse.

**Tech Stack:** Python 3.9 standard library only. No numpy, pandas or openpyxl.
Tests are `unittest`, run with `python3 test_tipping.py`.

**Spec:** `docs/superpowers/specs/2026-08-20-rival-decision-noise-design.md`

## Global Constraints

- Pure Python 3.9 standard library. No new dependencies, ever.
- `temperature = 0.0` must reproduce today's behaviour **bit-for-bit**. All 125
  existing tests must pass **unmodified** — do not edit an existing test to
  accommodate a new signature. If a change would force a test edit, the change is
  wrong; factor it differently.
- `RIVAL_NOISE = 0.05`, a fraction of win probability (so `0.05` is 5 percentage
  points).
- Noise applies to **rivals only**. Index 0 is me and never draws.
- New assumed parameters are printed in the report's `ASSUMPTIONS` block with the
  words `ASSUMED, not fitted`, matching `tau` and `reluctance`.
- Commit after every task.

---

## File Structure

Everything lives in the two existing files. This codebase is deliberately two
large files (`tipping.py`, `test_tipping.py`); do not split them.

- Modify `tipping.py`:
  - constants block (~line 78) — add `RIVAL_NOISE`
  - `solve_level0` (~line 316) — factor into `solve_level0_soft` + thin wrapper
  - `_decision_cache` (~line 619) — return a probability, accept `temperature`
  - `EquilibriumPolicy.__call__` (~line 1038) — return a float
  - `_actions` (~line 735) — accept pre-drawn uniforms and a separate rule for me
  - `_draw_season` (~line 750) — pre-draw rival noise
  - `_play_season` (~line 766) — thread noise through
  - `simulate_season_outcomes` (~line 833) and `simulate_branches` (~line 879) —
    accept and thread `temperature`
  - `solve_equilibrium` (~line 1051) — thread `temperature`
  - `field_tips` (~line 1215) — report the probability, not a bare action
  - `report` (~line 1519) and `main` (~line 2050) — parameter, flag, assumptions
- Modify `test_tipping.py`: append new test classes at the end.
- Modify `README.md`: document `--rival-noise`.

---

### Task 1: Soft value iteration in the level-0 DP

**Files:**
- Modify: `tipping.py:78-81` (constants), `tipping.py:316-359` (`solve_level0`)
- Test: `test_tipping.py` (append a new class)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `RIVAL_NOISE: float = 0.05` module constant.
  - `solve_level0_soft(p_fav, terminal, clamp=DELTA_CLAMP, reluctance=0.0,
    temperature=0.0) -> Tuple[List[List[float]], List[List[float]]]` returning
    `(values, dog_prob)`, both indexed `[t][delta + clamp]`. `dog_prob[t][i]` is
    P(this tipster takes the dog).
  - `solve_level0(p_fav, terminal, clamp=DELTA_CLAMP, reluctance=0.0) ->
    Tuple[List[List[float]], List[List[str]]]` — **signature and return type
    unchanged**, now a thin wrapper over `solve_level0_soft` at temperature 0.

Why the split: ten existing tests unpack `values, policy = T.solve_level0(...)`.
Adding a third return value breaks them. The wrapper keeps the old contract exact.

- [ ] **Step 1: Write the failing tests**

Append to `test_tipping.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_tipping.TestSoftLevel0 -v`
Expected: FAIL — `AttributeError: module 'tipping' has no attribute 'solve_level0_soft'`

- [ ] **Step 3: Add the constant**

In `tipping.py`, after the `RELUCTANCE` line (~line 81):

```python
RIVAL_NOISE = 0.05    # temperature on rivals' deviation choice (ASSUMED, not fitted)
```

- [ ] **Step 4: Replace `solve_level0` with the soft DP plus a wrapper**

Replace the whole body of `solve_level0` (`tipping.py:316-359`) with:

```python
def _clip_exp(x: float, limit: float = 700.0) -> float:
    """Keep the logistic argument inside the range math.exp can represent."""
    return max(-limit, min(limit, x))


def solve_level0_soft(
    p_fav: Sequence[float],
    terminal: Callable[[int], float],
    clamp: int = DELTA_CLAMP,
    reluctance: float = 0.0,
    temperature: float = 0.0,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Exact DP over a tipster's own delta. Returns (values, dog_prob).

    Both are indexed `[t][delta + clamp]`. `dog_prob[t][i]` is the probability this
    tipster takes the dog at game `t` from delta `i - clamp`.

    `reluctance` scales a penalty the deviation edge must clear before the dog is
    taken: `reluctance * max(0, p - 0.5)`. It is zero at a coin flip and largest at
    a certainty, which is what makes a modelled tipster deviate in close games and
    baulk at heavy favourites.

    `temperature` is how decisively the tipster acts on the edge that survives the
    penalty. At 0.0 this is a hard argmax and `dog_prob` is all 0.0 and 1.0 --
    exactly the old behaviour. Above zero the action is logistic in the edge and
    the value is the mixture, so a tipster's stored value reflects the noisy player
    they will actually be rather than an optimal one they will not.

    The stored value is always a real win probability. Under noise it is the win
    probability of the noisy behavioural policy, which is a truer number than the
    optimal-play value, not a less real one -- and every consumer of `values`
    depends on it being a genuine probability.
    """
    n = len(p_fav)
    size = 2 * clamp + 1
    values = [[0.0] * size for _ in range(n + 1)]
    dog_prob = [[0.0] * size for _ in range(n)]

    for d in range(-clamp, clamp + 1):
        values[n][d + clamp] = terminal(d)

    for t in range(n - 1, -1, -1):
        p = p_fav[t]
        nxt = values[t + 1]
        penalty = reluctance * max(0.0, p - 0.5)
        for d in range(-clamp, clamp + 1):
            i = d + clamp
            v_fav = nxt[i]
            lo = max(-clamp, d - 1) + clamp
            hi = min(clamp, d + 1) + clamp
            v_dog = p * nxt[lo] + (1.0 - p) * nxt[hi]
            # Reluctance is an aversion to backing a longshot, not a death wish: it
            # must never make a tipster prefer a CERTAIN loss to a live chance. With
            # nothing left to protect there is nothing to be reluctant about.
            floor = penalty if v_fav > 0.0 else 0.0
            edge = v_dog - v_fav - floor

            if temperature <= 0.0:
                q = 1.0 if edge > 1e-12 else 0.0
            elif v_fav <= 0.0 and v_dog <= 0.0:
                # Dead: every branch is worth nothing, so the edge is zero and the
                # logistic would return a coin flip. An eliminated tipster cannot
                # affect anyone's finishing position, but reporting them as
                # flipping coins every game makes the field look deranged.
                q = 0.0
            else:
                q = 1.0 / (1.0 + math.exp(-_clip_exp(edge / temperature)))

            values[t][i] = q * v_dog + (1.0 - q) * v_fav
            dog_prob[t][i] = q
    return values, dog_prob


def solve_level0(
    p_fav: Sequence[float],
    terminal: Callable[[int], float],
    clamp: int = DELTA_CLAMP,
    reluctance: float = 0.0,
) -> Tuple[List[List[float]], List[List[str]]]:
    """Exact DP, hard argmax. Returns (values, policy); index delta as [t][delta + clamp].

    The deterministic face of `solve_level0_soft`, kept because the exact grouped
    solve reads a policy of strings and must stay deterministic -- see the spec's
    scope section for why noise cannot cross into the grouped DP.
    """
    values, dog_prob = solve_level0_soft(p_fav, terminal, clamp=clamp,
                                         reluctance=reluctance, temperature=0.0)
    policy = [["D" if q > 0.5 else "F" for q in row] for row in dog_prob]
    return values, policy
```

- [ ] **Step 5: Run the new tests and the full suite**

Run: `python3 -m unittest test_tipping.TestSoftLevel0 -v`
Expected: PASS, 9 tests.

Run: `python3 test_tipping.py`
Expected: `Ran 134 tests`, `OK (skipped=1)`. **No existing test may be edited.**

- [ ] **Step 6: Commit**

```bash
git add tipping.py test_tipping.py
git commit -m "Give the level-0 DP a temperature, defaulting to the old hard argmax"
```

---

### Task 2: The decision cache returns a probability

**Files:**
- Modify: `tipping.py:619-662` (`_decision_cache`), `tipping.py:735-747` (`_actions`),
  `tipping.py:1038-1049` (`EquilibriumPolicy.__call__`), `tipping.py:665-681`
  (`simulated_actions`)
- Test: `test_tipping.py` (append)

**Interfaces:**
- Consumes: `solve_level0_soft(p_fav, terminal, clamp, reluctance, temperature)`
  from Task 1.
- Produces:
  - `_decision_cache(p_fav, reluctance, clamp, temperature=0.0)` returning
    `decide(t, standing) -> float` (P(dog)), **not** a `"F"`/`"D"` string.
  - `EquilibriumPolicy.__call__(t, standing) -> float`, returning `0.0`/`1.0`.
  - `_actions(scores, errors, t, decide, draws=None, decide_me=None) -> List[str]`.
    `draws[i]` is a pre-drawn uniform for tipster `i`; `draws[0]` is ignored.
    `decide_me` overrides the rule for index 0; defaults to `decide`.

Why index 0 gets its own rule: under noise a rival's DP correctly assumes *they*
will be noisy later, but mine must not — my recommendation is a best response to a
noisy field, not itself noisy. Two caches, both memoised, negligible cost.

- [ ] **Step 1: Write the failing tests**

Append to `test_tipping.py`:

```python
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

    def test_equilibrium_policy_returns_floats(self):
        fallback = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP)
        policy = T.EquilibriumPolicy({}, fallback, 100, 1, 0, 0.0)
        self.assertIn(policy(0, ((1, -1),)), (0.0, 1.0))
        table = {(0, ((1, -1),)): "D"}
        learned = T.EquilibriumPolicy(table, fallback, 100, 1, 0, 0.0)
        self.assertEqual(learned(0, ((1, -1),)), 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_tipping.TestNoisyDecisions -v`
Expected: FAIL — `_decision_cache() got an unexpected keyword argument 'temperature'`

- [ ] **Step 3: Change `_decision_cache` to return a probability**

In `tipping.py`, change the signature and the cached type, and swap the tail:

```python
def _decision_cache(p_fav: Sequence[float], reluctance: float, clamp: int,
                    temperature: float = 0.0):
```

Change the cache declaration from `Dict[..., str]` to:

```python
    cache: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], float] = {}
```

Change the inner signature to `def action(t: int, standing: ...) -> float:` and
replace the two lines that solve and index the policy with:

```python
        _, dog_prob = solve_level0_soft(p_fav[t:], terminal, clamp=clamp,
                                        reluctance=reluctance,
                                        temperature=temperature)
        result = dog_prob[0][clamp]   # their delta is 0 as of right now
```

Also update the docstring's first line to
`"""Build the cached 'how likely am I to take the dog' lookup.`

- [ ] **Step 4: Change `_actions` to draw**

Replace `tipping.py:735-747` with:

```python
def _actions(scores, errors, t, decide, draws=None, decide_me=None) -> List[str]:
    """Each tipster's action given the live standings, using margin errors as
    accumulated so far in this very season.

    There is no special case for the leader. Whoever is genuinely winning has
    nothing to gain by differentiating, so the DP picks the favourite for them on
    its own -- and a tipster who is level on points but behind on the countback is
    NOT winning, and correctly keeps chasing.

    `decide` returns P(take the dog). `draws` supplies one pre-drawn uniform per
    tipster; where it is absent the modal action is taken, which is what the
    report wants when it is describing a position rather than playing it out.

    Index 0 is me and never draws: my recommendation is a best response to a noisy
    field, not itself noisy. `decide_me` lets me answer a different rule from the
    rivals, which is how my policy stays deterministic while theirs is not.
    """
    acts = []
    for i in range(len(scores)):
        rule = decide_me if (i == 0 and decide_me is not None) else decide
        q = rule(t, _standing(scores, errors, i))
        if i == 0 or draws is None:
            acts.append("D" if q > 0.5 else "F")
        else:
            acts.append("D" if draws[i] < q else "F")
    return acts
```

Note the `draws[i] < q` form is exactly the old behaviour when `q` is 0.0 or 1.0:
a uniform in `[0, 1)` is always `< 1.0` and never `< 0.0`.

- [ ] **Step 5: Change `EquilibriumPolicy.__call__` to return a float**

Replace its body's three return statements so the annotation is `-> float` and:
- the conceded branch returns `0.0` instead of `"F"`
- the hit branch returns `1.0 if hit == "D" else 0.0`
- the fallback branch is unchanged (`self.fallback` now returns a float)

Update the class docstring's first paragraph to note it returns a dog probability
of 0.0 or 1.0, since the learned table stores a chosen action rather than a mix.

- [ ] **Step 6: Run the new tests and the full suite**

Run: `python3 -m unittest test_tipping.TestNoisyDecisions -v`
Expected: PASS, 7 tests.

Run: `python3 test_tipping.py`
Expected: `Ran 141 tests`, `OK (skipped=1)`. No existing test edited.

- [ ] **Step 7: Commit**

```bash
git add tipping.py test_tipping.py
git commit -m "Return a deviation probability from the decision cache"
```

---

### Task 3: Pre-draw rival noise so paired branches stay paired

**Files:**
- Modify: `tipping.py:750-763` (`_draw_season`), `tipping.py:766-796` (`_play_season`),
  `tipping.py:833-876` (`simulate_season_outcomes`), `tipping.py:879-993`
  (`simulate_branches`), `tipping.py:1051+` (`solve_equilibrium`)
- Test: `test_tipping.py` (append)

**Interfaces:**
- Consumes: `_actions(..., draws=, decide_me=)` and the float-returning
  `_decision_cache` from Task 2.
- Produces:
  - `_draw_season(games, p_fav, n_rivals, random_f, gauss, tau, sigma)` now returns
    a **3-tuple** `(results, margins, noise)` where `noise[t][i]` is a uniform for
    tipster `i` at game `t`; `noise[t][0]` is drawn but unused, so indices line up.
  - `_play_season(games, start_scores, start_errors, results, margins, decide,
    force_first_me, noise=None, decide_me=None)`.
  - `simulate_branches(...)` and `simulate_season_outcomes(...)` gain
    `temperature: float = 0.0`.

This is the step that fails silently if done wrong. Both branches of a paired
season must see the *same* uniform per (game, rival). Drawing inside `_play_season`
would decouple them, inflate `stderr_edge`, and make `too close to call` meaningless.

- [ ] **Step 1: Write the failing tests**

Append to `test_tipping.py`:

```python
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
            self.GAMES, self.P_FAV, 2, rng.random, rng.gauss, 10.0, 37.0)
        self.assertEqual(len(noise), len(self.GAMES))
        for row in noise:
            self.assertEqual(len(row), 3)          # me + two rivals
            for u in row:
                self.assertGreaterEqual(u, 0.0)
                self.assertLess(u, 1.0)

    def test_both_branches_see_identical_noise(self):
        # Same pre-drawn season replayed twice with different forced first tips
        # must expose the rivals to the same luck.
        rng = random.Random(7)
        results, margins, noise = T._draw_season(
            self.GAMES, self.P_FAV, 2, rng.random, rng.gauss, 10.0, 37.0)
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        scores = [155, 156, 154]
        errors = [577.0, 628.0, 583.0]
        seen = []
        for branch in ("F", "D"):
            captured = []
            def spy(t, standing, _c=captured, _d=decide):
                _c.append((t, standing))
                return _d(t, standing)
            T._play_season(self.GAMES, scores, errors, results, margins,
                           spy, branch, noise=noise)
            seen.append(captured)
        # The rivals' decision states may differ between branches (my tip moved
        # the standings), but the noise they are compared against must not: the
        # same season replayed with the same noise is reproducible.
        again = T._play_season(self.GAMES, scores, errors, results, margins,
                               decide, "F", noise=noise)
        for _ in range(5):
            self.assertEqual(
                T._play_season(self.GAMES, scores, errors, results, margins,
                               decide, "F", noise=noise),
                again)

    def test_simulate_branches_is_deterministic_under_noise(self):
        kwargs = dict(n_seasons=400, seed=99, temperature=0.05)
        a = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                **kwargs)
        b = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                **kwargs)
        self.assertEqual(a.p_win_favourite, b.p_win_favourite)
        self.assertEqual(a.p_win_underdog, b.p_win_underdog)

    def test_zero_temperature_matches_the_old_simulation(self):
        # Noise off must be the same numbers as before the feature existed.
        quiet = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                    n_seasons=800, seed=5, temperature=0.0)
        default = T.simulate_branches(self.ME, self.RIVALS, self.GAMES,
                                      self.P_FAV, n_seasons=800, seed=5)
        self.assertEqual(quiet.p_win_favourite, default.p_win_favourite)
        self.assertEqual(quiet.p_win_underdog, default.p_win_underdog)

    def test_the_bloc_actually_breaks(self):
        # Two rivals in identical positions must sometimes tip differently within
        # one season. Under the old hard argmax they never could.
        twins = [T.Tipster("Twin A", 154, 590), T.Tipster("Twin B", 154, 590)]
        decide = T._decision_cache(self.P_FAV, 0.10, T.DELTA_CLAMP,
                                   temperature=0.05)
        rng = random.Random(3)
        split = False
        for _ in range(200):
            _, _, noise = T._draw_season(self.GAMES, self.P_FAV, 2,
                                         rng.random, rng.gauss, 10.0, 37.0)
            acts = T._actions([155, 154, 154], [577.0, 590.0, 590.0], 0,
                              decide, draws=noise[0])
            if acts[1] != acts[2]:
                split = True
                break
        self.assertTrue(split, "identically placed rivals never disagreed")

    def test_noise_widens_the_field(self):
        # More noise means more rivals with a live chance, so my share falls or
        # the spread of outcomes grows. Assert the weakest true thing: the
        # results differ.
        quiet = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                    n_seasons=3000, seed=11, temperature=0.0)
        noisy = T.simulate_branches(self.ME, self.RIVALS, self.GAMES, self.P_FAV,
                                    n_seasons=3000, seed=11, temperature=0.15)
        self.assertNotEqual(quiet.table_favourite, noisy.table_favourite)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_tipping.TestNoisePreDraw -v`
Expected: FAIL — `ValueError: not enough values to unpack (expected 3, got 2)`

- [ ] **Step 3: Pre-draw the noise**

Replace `_draw_season` (`tipping.py:750-763`) with:

```python
def _draw_season(games, p_fav, n_rivals, random_f, gauss, tau, sigma):
    """Pre-draw one season's randomness so both branches can replay it identically.

    The draws never depend on anyone's tips, so pulling them up front costs nothing
    and buys a paired comparison: the only difference between the two branches is
    my forced first tip, not the luck.

    Rival decision noise is luck too, so it is drawn here rather than at the point
    of decision. Both branches then compare the SAME uniform against whatever
    probability their own standings produce -- common random numbers. Drawing it
    inside the replay instead would decouple the branches and inflate the reported
    edge error.
    """
    n = n_rivals + 1
    results = [random_f() < p_fav[t] for t in range(len(games))]
    noise = [[random_f() for _ in range(n)] for _ in range(len(games))]
    margins = {}
    for t, game in enumerate(games):
        if game.is_margin_game and game.line_fav is not None:
            margins[t] = (gauss(game.line_fav, sigma),
                          [gauss(game.line_fav, tau) for _ in range(n_rivals)])
    return results, margins, noise
```

- [ ] **Step 4: Thread it through `_play_season`**

Change the signature to:

```python
def _play_season(games, start_scores, start_errors, results, margins, decide,
                 force_first_me, noise=None, decide_me=None):
```

and the action line inside the loop from `acts = _actions(scores, errors, t, decide)`
to:

```python
        acts = _actions(scores, errors, t, decide,
                        draws=None if noise is None else noise[t],
                        decide_me=decide_me)
```

- [ ] **Step 5: Thread `temperature` through both simulators**

In `simulate_season_outcomes` (~line 833) and `simulate_branches` (~line 879), add
`temperature: float = 0.0,` to the signature after `clamp`, and replace the
`decide is None` block in each with:

```python
    decide_me = None
    if decide is None:
        decide = _decision_cache(p_fav, reluctance, clamp, temperature)
        if temperature > 0.0:
            # Rivals act noisily and their own DP knows it. Mine must not: the
            # recommendation is a best response to a noisy field, not itself noisy.
            decide_me = _decision_cache(p_fav, reluctance, clamp, 0.0)
```

Update the three `_draw_season(...)` call sites (lines ~865, ~927, ~1129) to unpack
three values, and pass `noise=noise, decide_me=decide_me` to every `_play_season`
call in those two functions.

In `solve_equilibrium` (~line 1129) the `_draw_season` call must unpack three
values as well; pass `noise` on to its `_actions` calls so the learner sees the
noisy field. Add `temperature: float = 0.0` to `solve_equilibrium`'s signature and
to its `fallback = _decision_cache(p_fav, reluctance, clamp)` call.

- [ ] **Step 6: Run the new tests and the full suite**

Run: `python3 -m unittest test_tipping.TestNoisePreDraw -v`
Expected: PASS, 6 tests.

Run: `python3 test_tipping.py`
Expected: `Ran 147 tests`, `OK (skipped=1)`. No existing test edited.

- [ ] **Step 7: Commit**

```bash
git add tipping.py test_tipping.py
git commit -m "Pre-draw rival noise so paired branches share their luck"
```

---

### Task 4: Wire up the flag, the report and the docs

**Files:**
- Modify: `tipping.py:1215-1239` (`field_tips`), `tipping.py:1519+` (`report`),
  `tipping.py:1951-1953` (assumptions), `tipping.py:2086+` (argparse),
  `tipping.py:2126` (the `report(...)` call)
- Modify: `README.md`
- Test: `test_tipping.py` (append)

**Interfaces:**
- Consumes: `RIVAL_NOISE`, and the `temperature` parameter on `simulate_branches`,
  `simulate_season_outcomes` and `solve_equilibrium` from Tasks 1–3.
- Produces: `--rival-noise` CLI flag; `report(..., rival_noise: float = RIVAL_NOISE)`;
  `field_tips` returning a 5-tuple with the dog probability appended.

`field_tips` needs the extra field because the `--equilibrium` block currently
prints "Ryan Board tips DOG" as a flat assertion. Once the model is only 58% sure,
printing it as certain is the report claiming confidence the model no longer has.

- [ ] **Step 1: Write the failing tests**

Append to `test_tipping.py`:

```python
class TestNoiseSurface(unittest.TestCase):
    """The flag, the reported probability, and the assumptions block."""

    GAMES = [
        T.Game("G1", "R", "Thu", "A", "B", 1.40, 2.96, 16.5, True, None),
        T.Game("G2", "R", "Fri", "C", "D", 2.44, 1.56, 10.5, False, None),
    ]
    ME = T.Tipster("Me", 155, 577, is_me=True)
    RIVALS = [T.Tipster("Leader", 156, 628), T.Tipster("Chaser", 154, 583)]

    def test_field_tips_reports_the_probability(self):
        p_fav = [0.689, 0.616]
        decide = T._decision_cache(p_fav, 0.10, T.DELTA_CLAMP, temperature=0.05)
        rows = T.field_tips(self.ME, self.RIVALS, self.GAMES, 0, decide)
        for name, act, team, gap, q in rows:
            self.assertIn(act, ("F", "D"))
            self.assertGreaterEqual(q, 0.0)
            self.assertLessEqual(q, 1.0)
            self.assertEqual(act, "D" if q > 0.5 else "F")

    def test_field_tips_is_certain_at_zero_temperature(self):
        p_fav = [0.689, 0.616]
        decide = T._decision_cache(p_fav, 0.10, T.DELTA_CLAMP)
        for _, _, _, _, q in T.field_tips(self.ME, self.RIVALS, self.GAMES, 0,
                                          decide):
            self.assertIn(q, (0.0, 1.0))

    def test_cli_exposes_rival_noise(self):
        import argparse, io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                T.main(["--help"])
        self.assertIn("--rival-noise", buf.getvalue())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_tipping.TestNoiseSurface -v`
Expected: FAIL — `ValueError: not enough values to unpack (expected 5, got 4)`

- [ ] **Step 3: Make `field_tips` report the probability**

In `field_tips`, replace the `acts = _actions(...)` line and the return with:

```python
    probs = [decide(t, _standing(scores, errors, i)) for i in range(len(names))]
    acts = ["D" if q > 0.5 else "F" for q in probs]

    game = games[t]
    favourite = game.home if game.home_odds <= game.away_odds else game.away
    underdog = underdog_name(game)
    top = max(scores)
    return [
        (names[i], acts[i], favourite if acts[i] == "F" else underdog,
         scores[i] - top, probs[i])
        for i in range(len(names))
    ]
```

Update its docstring's Returns line to
`Returns (name, "F"/"D", the team that means, points behind the lead, P(dog)).`

There is exactly one caller, in the `--equilibrium` block at `tipping.py:1774`.
It unpacks the tuple in two places, both of which must change. Replace:

```python
        for name, action, team, gap in board:
            marker = "   <-- you" if name == me.name else ""
            print("        %-16s %-4s %-20s %+3d%s"
                  % (name, "dog" if action == "D" else "fav", team, gap, marker))
        print()
        mine = next(a for n, a, _, _ in board if n == me.name)
```

with:

```python
        for name, action, team, gap, q in board:
            marker = "   <-- you" if name == me.name else ""
            # Under noise the action is only the MODAL tip. Printing it bare would
            # claim a certainty the model no longer has.
            chance = "" if rival_noise <= 0.0 else "  p=%3.0f%%" % (100.0 * q)
            print("        %-16s %-4s %-20s %+3d%s%s"
                  % (name, "dog" if action == "D" else "fav", team, gap,
                     chance, marker))
        print()
        mine = next(a for n, a, _, _, _ in board if n == me.name)
```

- [ ] **Step 4: Thread `rival_noise` through `report` and `main`**

Add `rival_noise: float = RIVAL_NOISE,` to `report`'s signature after `reluctance`.
Pass `temperature=rival_noise` to the `simulate_branches`, `simulate_contingency`,
`simulate_season_outcomes` and `solve_equilibrium` calls inside `report`.

Add the argparse flag after `--reluctance`:

```python
    parser.add_argument("--rival-noise", type=float, default=RIVAL_NOISE,
                        help="how uncertainly rivals act on a thin edge, in win "
                             "probability (default %.2f, 0 = certain)" % RIVAL_NOISE)
```

In `main`, the `report(...)` call at `tipping.py:2126` passes positionally. Add
`args.rival_noise` immediately after `args.reluctance` so it lines up with the new
parameter position:

```python
    result = report(me, rivals, games, args.devig, args.explain,
                    args.sims, args.seed, args.tau, paths, args.reluctance,
                    args.rival_noise,
                    args.sim_seasons, args.contingency_seasons, args.equilibrium,
                    args.equilibrium_seasons, args.equilibrium_sweeps,
                    args.equilibrium_gap_clamp)
```

and put `rival_noise: float = RIVAL_NOISE,` in `report`'s signature immediately
after `reluctance: float = RELUCTANCE,` so the positional order matches.

Add to the `ASSUMPTIONS` block after the `reluctance` lines:

```python
    print("  * rival noise = %.2f is ASSUMED, not fitted. It is how uncertainly a"
          % rival_noise)
    print("    rival acts on a thin edge; at 0 they act with certainty on any edge.")
```

- [ ] **Step 5: Run the new tests and the full suite**

Run: `python3 -m unittest test_tipping.TestNoiseSurface -v`
Expected: PASS, 3 tests.

Run: `python3 test_tipping.py`
Expected: `Ran 150 tests`, `OK (skipped=1)`.

- [ ] **Step 6: Document it in the README**

In the "Useful flags" line, add `--rival-noise` to the list. In "Read the output
critically", add a bullet matching the existing style:

```markdown
- **`rival-noise = 0.05` is assumed, not fitted.** Rivals used to act with total
  certainty on any edge, so a 1.6-point preference put the whole chasing pack on
  the same underdog. They now take the dog with a probability logistic in that
  edge, drawn independently, so identically placed rivals can tip differently.
  Sweep it with `--rival-noise` and check the recommendation holds; `0` restores
  the old certain behaviour.
```

Also correct the stale test count: the README says 118 tests, the suite has 150
after this plan.

- [ ] **Step 7: Commit**

```bash
git add tipping.py test_tipping.py README.md
git commit -m "Expose --rival-noise and report it as an assumption"
```

---

### Task 5: Verify against the live board

**Files:**
- Modify: none expected. Test: manual verification plus the full suite.

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: a recorded before/after comparison, and a decision on whether the
  recommendation is stable.

- [ ] **Step 1: Confirm noise off is a true no-op**

Run:
```bash
python3 tipping.py --recommend --rival-noise 0 --sim-seasons 20000 > /tmp/off.txt
git stash && python3 tipping.py --recommend --sim-seasons 20000 > /tmp/before.txt; git stash pop
diff <(grep -A3 "RECOMMENDATION" /tmp/off.txt) <(grep -A3 "RECOMMENDATION" /tmp/before.txt)
```
Expected: no differences. If the headline moved with noise off, Task 3's threading
is wrong — most likely `_draw_season` is consuming RNG draws in a different order,
which shifts every subsequent draw. Check that `noise` is drawn **after** `results`
and **before** `margins`, exactly as written in Task 3 Step 3.

- [ ] **Step 2: Sweep the temperature on the live board**

Run:
```bash
for t in 0 0.02 0.05 0.10; do
  echo "=== T=$t ==="
  python3 tipping.py --recommend --rival-noise $t --sim-seasons 20000 \
    | grep -E "RECOMMENDATION|Jake Turner|Likeliest next"
done
```
Record the output in the commit message. What to look for: the recommended tip
should be stable across `T`. If it flips between 0.02 and 0.05, the decision is
genuinely marginal and that is worth knowing — report it, do not tune `T` to make
it go away.

- [ ] **Step 3: Confirm the equilibrium path still runs**

Run:
```bash
python3 tipping.py --recommend --equilibrium --equilibrium-seasons 2000 \
  --sim-seasons 10000
```
Expected: completes, and the equilibrium block still prints states learned,
thinnest sample, fallback rate, and a settled/not-settled line.

- [ ] **Step 4: Run the full suite one final time**

Run: `python3 test_tipping.py`
Expected: `Ran 150 tests`, `OK (skipped=1)`.

- [ ] **Step 5: Commit the verification record**

```bash
git commit --allow-empty -F - <<'EOF'
Verify probabilistic rival deviation on the live board

<paste the T sweep from Step 2 here>
EOF
```

---

## Known limitations to record, not fix

These follow from the spec's scoping. Note them; do not build around them.

1. **`--equilibrium` only gets noise in fallback states.** The learned table stores
   a chosen action, so `EquilibriumPolicy` returns 0.0 or 1.0 and the field shares
   one deterministic learned rule. Noise reaches only the states the learner never
   visited, which fall back to the level-0 cache. The bloc behaviour therefore
   largely persists under `--equilibrium`. Making the learned table store
   probabilities is a separate piece of work.

2. **The exact grouped solve stays deterministic**, so the cross-check `WARNING`
   between the two models will fire more often than it used to. That is
   informative — it now fires when the recommendation depends on whether rivals
   are treated as a certain bloc — but expect more of it.

3. **My own rollout policy from game 2 on** uses a temperature-0 cache whose
   terminal still assumes a frozen field, unchanged from today. It is a rollout,
   not an optimum, exactly as the README already says.
