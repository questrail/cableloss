# CHANGELOG.md

This file contains all notable changes to the [cableloss][] project.

## Unreleased

## v0.3.0 - 2026-09-01

### Added

- Build, lock, and run the project with [uv][] and [Just][]. The `setup.py`,
  `setup.cfg`, `MANIFEST.in`, `requirements.txt`, `unittest.cfg`, and Invoke
  `tasks.py` are gone, replaced by a `pyproject.toml` with a hatchling build
  backend, a `uv.lock` that pins numpy and the dev tools exactly, and a
  `Justfile` carrying the recipes. `requirements.txt` had pinned a
  development environment and a runtime dependency in one list with nothing
  distinguishing them, and it recorded no hashes and no transitive versions,
  so nothing reproduced an environment from it.
- Run CI on GitHub Actions across Python 3.12, 3.13, and 3.14. The README
  still advertised a Travis CI badge for a build that had not run since
  Travis stopped serving open source repositories, so the badge was the only
  thing standing where the checks used to be. Every push now lints with
  [ruff][], type checks with [pyright][], and runs the suite on each
  interpreter, and reports coverage to Coveralls from one leg of the matrix.
- Test the dependency floor in CI. The `numpy>=2.2.0` in `pyproject.toml` is
  a promise to anyone installing cableloss alongside something that holds
  numpy back, and the matrix never tests it: it installs whatever `uv.lock`
  pins, which is the newest numpy rather than the oldest allowed one. A
  separate job resolves with `--resolution lowest-direct` so that a change
  which quietly starts needing a newer numpy fails there rather than in a
  bug report.
- Audit the workflows with [zizmor][] in CI and in `just lint`. Everything
  under `src/` is linted on every push; the workflows, which are the part of
  this repository that can mint a PyPI credential, would otherwise be read by
  eye alone. It runs as a job of its own rather than as a step gated on one
  leg of the matrix, since that gate would mean dropping a Python version
  silently stops the audit.
- Publish to PyPI from a tag with [trusted publishing][], and sign a
  [PEP 740][] attestation for each distribution. Releases had gone out by
  hand, `python -m twine upload dist/*` against a `.pypirc` holding a
  username and password in plaintext on disk. The release workflow mints a
  short lived credential from the GitHub OIDC identity of the run instead,
  so there is no token to store, rotate, or leak, and the same identity
  signs an attestation that PyPI serves beside the file it attests. Before
  it uploads it waits on the whole CI workflow, checks that the tagged
  commit is on `master`, and checks the tag against the version in
  `pyproject.toml`.
- Smoke test the built wheel before publishing it. Every other check runs
  against the source tree, so a packaging mistake that leaves a module or
  `py.typed` out of the distribution passes ruff, pyright, and the whole
  suite and ships anyway. `scripts/smoke_test_wheel.py` installs the wheel
  where `src/` cannot be reached and imports it there. `just build` runs the
  same script, so a local build and the release check the same things.
- Cut releases with `just release`, which refuses a dirty tree, a branch
  other than `master`, a `master` behind its upstream, an empty Unreleased
  section, or a tag that already exists, then shows the entries waiting and
  the version each kind of bump would produce and asks which to cut. The old
  `invoke release` printed a checklist of questions to answer from memory and
  took the version as an argument.
- Put the actions and the Python dependencies under Dependabot. The actions
  in both workflows are pinned to commit SHAs rather than to mutable tags,
  since the release workflow can mint a PyPI credential; a pin with nothing
  updating it is a decision to stay on one commit forever.
- Ship `py.typed`, so that the annotations reach the type checkers of
  projects that depend on cableloss rather than stopping at this repository.
- Require 100% statement and branch coverage. The suite covers every line;
  a floor set anywhere below that would let coverage fall without anything
  noticing.

### Changed

- **Breaking:** dropped `cableloss.__version__`. The version now lives in
  `pyproject.toml` alone rather than in two places that could disagree. Read
  it with `importlib.metadata.version("cableloss")`.
- **Breaking:** require Python 3.12 or newer, and numpy 2.2 or newer. The
  classifiers had advertised 3.5 through 3.9, none of which have been
  supported upstream for years.
- Moved the module into a `src/cableloss/` package. `import cableloss` and
  `cableloss.loss()` are unchanged; the layout is what makes the wheel smoke
  test meaningful, since a flat module sits on the path whether or not it was
  packaged.
- Annotated `loss()` as returning `npt.NDArray`. It had been annotated
  `np.array`, which is the factory function rather than a type, so the
  annotation described the return as a callable and no checker could use it.
  The `Union[int, float]` on `length` is now `int | float`.
- Replaced nose2, pep8, and mypy with pytest, ruff, and pyright, and rewrote
  the suite in pytest style. It had asserted one cable type at one length;
  it now covers all four cables, the published figures at both ends of each
  table, the linear scaling with length, the sort order interpolating callers
  rely on, and the `KeyError` an unknown cable type raises.
- Renamed the local `loss` inside `loss()` to `cable_loss`, which had
  shadowed the function it was returned from.
- Dropped `AUTHORS.md`, which listed one author and no contributors and was
  pointed at by a `LICENSE.txt` line naming a file (`AUTHORS.txt`) that never
  existed. Git already records who wrote what.

## v0.2.1 - 2022-06-25

- Eliminate pandoc for long_description.

## v0.2.0 - 2022-06-25

- Eliminate Python 2.x support.
- Update dependencies

## v0.1.0 - 2017-07-24

### Added

- Ability to calculate the cable loss.

[cableloss]: https://github.com/questrail/cableloss
[just]: https://just.systems/
[PEP 740]: https://peps.python.org/pep-0740/
[pyright]: https://microsoft.github.io/pyright/
[ruff]: https://docs.astral.sh/ruff/
[trusted publishing]: https://docs.pypi.org/trusted-publishers/
[uv]: https://docs.astral.sh/uv/
[zizmor]: https://docs.zizmor.sh/
