# Named Input Sets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `tipping.py` run against named input sets — `current` (reality) and `scenario` (sandbox) — selected with one `--set` flag, with per-set outputs so a sandbox run can never overwrite or be mistaken for a real recommendation.

**Architecture:** A `SetPaths` dataclass plus a pure `resolve_set(name)` function become the single source of truth for path construction. Everything that touches the filesystem — the template writer, the loaders, the report banner, the CSV writer — takes paths from a `SetPaths` rather than from module constants. A `copy_set(source, dest)` helper clones reality into a sandbox. The solver, devig, rival model and countback are untouched.

**Tech Stack:** Python 3.9 standard library only — `argparse`, `csv`, `dataclasses`, `os`, `shutil`, `unittest`. No third-party dependencies may be introduced.

## Global Constraints

- Pure Python standard library. No numpy, pandas, openpyxl, pytest, or any other third-party package.
- Must run on system Python 3.9. No `match`, no `X | Y` unions at runtime, no `dataclasses.slots`.
- Tests live in `test_tipping.py`, use `unittest`, and run via `python3 test_tipping.py`.
- No change to the solver, devig, rival model, or countback. This plan touches path resolution, the CLI, and reporting labels only.
- `inputs/fixtures_PLACEHOLDER.csv` stays at `inputs/` root — it is format reference material, not a set.
- Default set name is `current`. Any other set name is treated as a sandbox.

---

### Task 1: `SetPaths` and `resolve_set`

**Files:**
- Modify: `tipping.py:29-34` (path constants)
- Test: `test_tipping.py` (new `TestInputSets` class, appended before `if __name__`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SetPaths` frozen dataclass with fields `name: str`, `leaderboard: str`, `fixtures: str`, `output_dir: str`. Function `resolve_set(name: str, input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> SetPaths`. Constant `DEFAULT_SET = "current"`. Tasks 2–6 all consume these.

- [ ] **Step 1: Write the failing test**

Append to `test_tipping.py`, immediately before the `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: FAIL with `AttributeError: module 'tipping' has no attribute 'resolve_set'`

- [ ] **Step 3: Write the implementation**

In `tipping.py`, replace lines 29-34 with:

```python
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
```

`InputError` is defined later in the file but `resolve_set` only references it at call time, so the forward reference is fine. `dataclass` and `Optional` are already imported at lines 25-27.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tipping.py test_tipping.py
git commit -m "Add SetPaths and resolve_set for named input sets"
```

---

### Task 2: Migrate the existing CSVs into `inputs/current/`

**Files:**
- Move: `inputs/leaderboard.csv` → `inputs/current/leaderboard.csv`
- Move: `inputs/fixtures.csv` → `inputs/current/fixtures.csv`

**Interfaces:**
- Consumes: `LEADERBOARD_CSV` / `FIXTURES_CSV` from Task 1, which now point at `inputs/current/`.
- Produces: a working `inputs/current/` set. Task 6 seeds `inputs/scenario/` from it.

This task has no test of its own — Task 1's tests already assert the constants point at `inputs/current/`, and the verification here is that the engine still runs end to end.

- [ ] **Step 1: Verify the engine is currently broken**

Run: `python3 tipping.py --recommend`
Expected: `INPUT ERROR: .../inputs/current/leaderboard.csv not found.` — Task 1 repointed the constants but the files have not moved yet.

- [ ] **Step 2: Move the files**

```bash
mkdir -p inputs/current
git mv inputs/leaderboard.csv inputs/current/leaderboard.csv
git mv inputs/fixtures.csv inputs/current/fixtures.csv
```

- [ ] **Step 3: Verify the engine runs again**

Run: `python3 tipping.py --recommend`
Expected: the full report, recommending FREMANTLE for R23G1. `inputs/fixtures_PLACEHOLDER.csv` must still be at `inputs/` root.

- [ ] **Step 4: Run the full suite**

Run: `python3 test_tipping.py`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add -A inputs
git commit -m "Move the live CSVs into the current input set"
```

---

### Task 3: Set-aware `write_template`

**Files:**
- Modify: `tipping.py:609-634` (`write_template`)
- Test: `test_tipping.py` (`TestInputSets`)

**Interfaces:**
- Consumes: `SetPaths`, `resolve_set` from Task 1.
- Produces: `write_template(paths: SetPaths) -> None` — signature change from the current no-argument form. Task 5 calls it from `main`.

- [ ] **Step 1: Write the failing test**

Add to `TestInputSets` in `test_tipping.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest test_tipping.TestInputSets.test_template_writes_into_the_named_set -v`
Expected: FAIL with `TypeError: write_template() takes 0 positional arguments but 1 was given`

- [ ] **Step 3: Write the implementation**

In `tipping.py`, change the `write_template` signature and its first lines. Replace lines 609-612:

```python
def write_template(paths: SetPaths) -> None:
    os.makedirs(os.path.dirname(paths.leaderboard), exist_ok=True)
    with open(paths.leaderboard, "w", newline="") as fh:
        csv.writer(fh).writerows(TEMPLATE_LEADERBOARD)
```

Replace lines 621-625 (the fixtures write and the two `print` lines):

```python
    with open(paths.fixtures, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    print("Wrote %s" % paths.leaderboard)
    print("Wrote %s" % paths.fixtures)
```

Then update the final line of the function (line 634) so the printed command names the set:

```python
    print("Delete rows for games already played. Then: python3 tipping.py --recommend --set %s"
          % paths.name)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tipping.py test_tipping.py
git commit -m "Make write_template write into a named set"
```

---

### Task 4: `copy_set`

**Files:**
- Modify: `tipping.py` (add `copy_set` immediately after `write_template`)
- Test: `test_tipping.py` (`TestInputSets`)

**Interfaces:**
- Consumes: `SetPaths`, `resolve_set` from Task 1; `write_template` from Task 3 (tests only, to build a source set).
- Produces: `copy_set(source: SetPaths, dest: SetPaths, force: bool = False, confirm: Callable[[str], str] = input) -> List[str]` returning the list of paths written, empty if the user declined. Task 5 calls it from `main`.

- [ ] **Step 1: Write the failing test**

Add to `TestInputSets` in `test_tipping.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: 6 FAIL with `AttributeError: module 'tipping' has no attribute 'copy_set'`

- [ ] **Step 3: Write the implementation**

Add `import shutil` to the import block in `tipping.py` (alphabetically, between `random` and `sys` at lines 23-24). Then add this function immediately after `write_template`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tipping.py test_tipping.py
git commit -m "Add copy_set to clone one input set over another"
```

---

### Task 5: Wire `--set`, `--copy-set` and `--force` into the CLI

**Files:**
- Modify: `tipping.py:1010-1047` (`main`)
- Modify: `tipping.py:11-14` (module docstring usage block)
- Test: `test_tipping.py` (`TestInputSets`)

**Interfaces:**
- Consumes: `resolve_set`, `DEFAULT_SET`, `SetPaths` (Task 1); `write_template(paths)` (Task 3); `copy_set(...)` (Task 4).
- Produces: `effective_paths(set_name, leaderboard, fixtures) -> Tuple[SetPaths, List[str]]` returning the resolved paths plus a list of warning strings. Task 6 relies on `--set` working.

- [ ] **Step 1: Write the failing test**

Add to `TestInputSets` in `test_tipping.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: 4 FAIL with `AttributeError: module 'tipping' has no attribute 'effective_paths'`

- [ ] **Step 3: Write `effective_paths`**

Add to `tipping.py` immediately after `copy_set`:

```python
def effective_paths(set_name: str,
                    leaderboard: Optional[str],
                    fixtures: Optional[str]) -> Tuple[SetPaths, List[str]]:
    """Resolve `set_name`, then apply any explicit per-file overrides.

    Overriding exactly one of the two pairs a leaderboard from one world with a
    fixture list from another, which is silently wrong rather than an error --
    so it returns a warning the caller must print.
    """
    paths = resolve_set(set_name)
    warnings: List[str] = []
    if leaderboard and not fixtures:
        warnings.append(
            "--leaderboard %s overrides set %r, but --fixtures does not: "
            "still reading fixtures from %s" % (leaderboard, set_name, paths.fixtures)
        )
    elif fixtures and not leaderboard:
        warnings.append(
            "--fixtures %s overrides set %r, but --leaderboard does not: "
            "still reading the leaderboard from %s"
            % (fixtures, set_name, paths.leaderboard)
        )
    return (
        SetPaths(
            name=paths.name,
            leaderboard=leaderboard or paths.leaderboard,
            fixtures=fixtures or paths.fixtures,
            output_dir=paths.output_dir,
        ),
        warnings,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: 15 tests PASS

- [ ] **Step 5: Rewrite `main` to use it**

Replace `tipping.py` lines 1013-1020 (the `--make-template` through `--leaderboard` argument definitions) with:

```python
    parser.add_argument("--make-template", action="store_true",
                        help="write blank CSVs into the target set")
    parser.add_argument("--recommend", action="store_true",
                        help="solve and print the next decision")
    parser.add_argument("--explain", action="store_true",
                        help="dump the reasoning for the next few games")
    parser.add_argument("--set", dest="set_name", default=DEFAULT_SET, metavar="NAME",
                        help="input set to read: inputs/NAME/ (default %s)" % DEFAULT_SET)
    parser.add_argument("--copy-set", metavar="NAME",
                        help="clone the %s set into inputs/NAME/, then exit" % DEFAULT_SET)
    parser.add_argument("--force", action="store_true",
                        help="with --copy-set, overwrite without prompting")
    parser.add_argument("--fixtures", default=None,
                        help="explicit fixtures CSV, overriding --set")
    parser.add_argument("--leaderboard", default=None,
                        help="explicit leaderboard CSV, overriding --set")
```

Then replace lines 1029-1047 (everything from `if args.make_template:` to `return 0`) with:

```python
    try:
        paths, warnings = effective_paths(args.set_name, args.leaderboard, args.fixtures)

        if args.copy_set:
            written = copy_set(resolve_set(DEFAULT_SET), resolve_set(args.copy_set),
                               force=args.force)
            if not written:
                print("Aborted. Nothing was written.")
                return 0
            for path in written:
                print("Wrote %s" % path)
            return 0

        if args.make_template:
            write_template(paths)
            return 0

        if not args.recommend:
            parser.print_help()
            return 0

        for warning in warnings:
            print("WARNING: %s" % warning, file=sys.stderr)

        me, rivals = load_leaderboard(paths.leaderboard)
        games = load_fixtures(paths.fixtures)
    except InputError as exc:
        print("INPUT ERROR: %s" % exc, file=sys.stderr)
        return 2

    result = report(me, rivals, games, args.devig, args.explain,
                    args.sims, args.seed, args.tau, paths)
    out = write_csv_outputs(me, rivals, games, result, args.devig, paths)
    print("Wrote %s" % out)
    return 0
```

`report` and `write_csv_outputs` do not accept `paths` yet — that is Task 6, so the suite will fail between these two steps. Complete Task 6 before committing this task.

- [ ] **Step 6: Verify the new flags parse**

Run: `python3 tipping.py --help`
Expected: `--set NAME`, `--copy-set NAME`, and `--force` all listed. Do not run `--recommend` yet; it will fail until Task 6 lands.

---

### Task 6: Set-aware reporting, outputs, and docs

**Files:**
- Modify: `tipping.py:737-746` (`report` signature), `tipping.py:773-781` (banner), `tipping.py:970-980` (`write_csv_outputs`)
- Modify: `README.md`
- Test: `test_tipping.py` (`TestInputSets`)

**Interfaces:**
- Consumes: `SetPaths`, `DEFAULT_SET` (Task 1); the `main` wiring (Task 5).
- Produces: `report(..., paths: SetPaths)` and `write_csv_outputs(..., paths: SetPaths)`. Terminal task — nothing consumes these.

- [ ] **Step 1: Write the failing test**

Add to `TestInputSets` in `test_tipping.py`:

```python
    def test_scenario_banner_fires_for_non_default_sets(self):
        self.assertIn("NOT REALITY", T.set_banner(T.resolve_set("scenario")))
        self.assertIn("scenario", T.set_banner(T.resolve_set("scenario")))

    def test_no_scenario_banner_for_the_current_set(self):
        self.assertNotIn("NOT REALITY", T.set_banner(T.resolve_set(T.DEFAULT_SET)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: 2 FAIL with `AttributeError: module 'tipping' has no attribute 'set_banner'`

- [ ] **Step 3: Add `set_banner`**

Add to `tipping.py` immediately after the `rule` function (which ends at line 734):

```python
def set_banner(paths: SetPaths) -> str:
    """The header lines identifying which world this run describes."""
    lines = ["Input set   : %s" % paths.name,
             "  leaderboard: %s" % paths.leaderboard,
             "  fixtures   : %s" % paths.fixtures]
    if paths.name != DEFAULT_SET:
        lines.insert(0, "*** SCENARIO SET -- NOT REALITY ***")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_tipping.TestInputSets -v`
Expected: 17 tests PASS

- [ ] **Step 5: Thread `paths` through `report`**

In `tipping.py`, add a parameter to the `report` signature — replace line 745 (`tau: float,`) with:

```python
    tau: float,
    paths: SetPaths,
```

Then print the banner in the header. Replace lines 773-777 with:

```python
    print()
    print(rule("="))
    print("AFL TIPPING -- NEXT DECISION")
    print(rule("="))
    print(set_banner(paths))
    print(rule())
    print("Game        : %s  %s v %s" % (g0.game_id, g0.home, g0.away))
```

- [ ] **Step 6: Thread `paths` through `write_csv_outputs`**

Replace lines 970-982 with:

```python
def write_csv_outputs(me: Tipster, rivals: List[Tipster], games: List[Game],
                      result: Dict[str, object], method: str, paths: SetPaths) -> str:
    os.makedirs(paths.output_dir, exist_ok=True)
    path = os.path.join(paths.output_dir, "recommendation.csv")
    sol: Solution = result["solution"]           # type: ignore[assignment]
    p_fav: List[float] = result["p_fav"]          # type: ignore[assignment]
    fav_names: List[str] = result["fav_names"]    # type: ignore[assignment]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value", "detail"])
        w.writerow(["set", "name", paths.name,
                    "REAL" if paths.name == DEFAULT_SET else "SCENARIO -- NOT REALITY"])
        w.writerow(["next", "game_id", games[0].game_id,
                    "%s v %s" % (games[0].home, games[0].away)])
        w.writerow(["next", "lock_local", games[0].lock_local, ""])
```

- [ ] **Step 7: Verify both sets run end to end**

```bash
python3 tipping.py --copy-set scenario --force
python3 tipping.py --recommend
python3 tipping.py --recommend --set scenario
```

Expected: the first writes both `inputs/scenario/` files. The second prints `Input set   : current` and writes `output/current/recommendation.csv`. The third prints `*** SCENARIO SET -- NOT REALITY ***` and writes `output/scenario/recommendation.csv`. Both recommend FREMANTLE, since the sets are still identical.

- [ ] **Step 8: Verify the mismatch warning and a bad set name**

```bash
python3 tipping.py --recommend --leaderboard inputs/current/leaderboard.csv
python3 tipping.py --recommend --set nope
```

Expected: the first prints a `WARNING:` naming the still-set-derived fixtures path, then runs normally. The second prints `INPUT ERROR: .../inputs/nope/leaderboard.csv not found` and exits 2.

- [ ] **Step 9: Run the full suite**

Run: `python3 test_tipping.py`
Expected: all tests PASS — the 35 originals plus 17 new.

- [ ] **Step 10: Update the README**

In `README.md`, replace the `## Use` code block with:

````markdown
```sh
python3 tipping.py --make-template            # write inputs/current/*.csv
# ... fill in inputs/current/fixtures.csv ...
python3 tipping.py --recommend                # the next decision
python3 tipping.py --recommend --explain      # + contingency table for 3 games
python3 test_tipping.py                       # tests, ~0.7s
```
````

Add this section immediately after the `## Use` section:

````markdown
## Input sets

Inputs live in named **sets**. A set is a directory pairing one leaderboard with
one fixture list, so the two can never be mixed across sets by accident.

```
inputs/current/     reality -- keep this matching the comp site
inputs/scenario/    sandbox -- edit freely to test what-ifs
output/current/     output/scenario/
```

```sh
python3 tipping.py --recommend                  # reads inputs/current/
python3 tipping.py --recommend --set scenario   # reads inputs/scenario/
python3 tipping.py --copy-set scenario          # reset the sandbox from current/
```

Set names are arbitrary, so `--set whatif-r23` works with no code change. Any set
other than `current` prints a `*** SCENARIO SET -- NOT REALITY ***` banner and
writes to its own output directory, so a sandbox run can never overwrite a real
recommendation or be mistaken for one afterwards.

`--fixtures` and `--leaderboard` still take explicit paths and override `--set`
for that one file. Overriding exactly one of the two prints a warning, because it
pairs a leaderboard from one world with fixtures from another.
````

Then update the `## Filling in` heading and its first line to name `inputs/current/fixtures.csv`, and update the `## Layout` block to:

````markdown
```
tipping.py         the engine (devig, countback, rival model, joint DP, report)
test_tipping.py    tests
inputs/current/    leaderboard.csv, fixtures.csv -- reality
inputs/scenario/   leaderboard.csv, fixtures.csv -- sandbox
output/<set>/      recommendation.csv
docs/superpowers/  specs/ and plans/
```
````

- [ ] **Step 11: Update the module docstring**

In `tipping.py`, replace lines 11-14 with:

```python
Usage:
    python3 tipping.py --make-template     # write inputs/current/*.csv to fill
    python3 tipping.py --recommend         # solve and print the next decision
    python3 tipping.py --recommend --explain
    python3 tipping.py --recommend --set scenario   # run against a sandbox set
    python3 tipping.py --copy-set scenario          # reset the sandbox
```

- [ ] **Step 12: Commit**

```bash
git add tipping.py test_tipping.py README.md inputs/scenario
git commit -m "Add --set, --copy-set and per-set outputs"
```

---

## Verification

After Task 6, all of the following must hold:

- `python3 test_tipping.py` passes every test.
- `python3 tipping.py --recommend` reads `inputs/current/` and writes `output/current/recommendation.csv`.
- `python3 tipping.py --recommend --set scenario` prints the NOT REALITY banner and writes `output/scenario/recommendation.csv`.
- `output/recommendation.csv` (the old un-setted path) is no longer written; delete the stale file and commit the deletion.
- `git status` is clean.
