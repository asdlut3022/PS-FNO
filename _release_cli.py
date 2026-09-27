"""Shared command line for the five benchmark entry points.

Why this exists
---------------
Each entry point used to hard-code its seed list and write its result JSON to
``<repo>/results/``. Two consequences:

* a seed that is documented but not in the hard-coded list (e.g. the wave
  seeds 314/777 behind ``results_wave/training_nt5_seed314_777.log``) could
  only be trained by editing or copying the program;
* a fresh run overwrote the archived record that backs the reported values.

``apply_cli`` adds ``--seed`` / ``--seeds``, ``--output-dir``, ``--ckpt-dir``,
``--device`` and a ``--tiny`` preset for CPU pipeline self-tests, and refuses,
by default, to overwrite a file that already exists in the shipped
``results/`` directory.

The result JSON and intermediate files routed through ``apply_cli`` are derived
from a single output root, so ``--output-dir D`` places those CLI-managed
artifacts under ``D`` (including ``D/checkpoints`` when the entry point uses the
CLI-managed checkpoint path). This guarantee does not cover
entry-point-specific or auxiliary outputs that resolve paths independently of
``apply_cli``.

After the run, ``apply_cli`` re-walks the package and reports any file that
appeared inside it during the run, making unintended package writes observable
rather than assuming that every output path is isolated.

Defaults are unchanged: with no arguments each script runs exactly the
configuration reported in the manuscript. ``--tiny`` is a smoke-test preset
only -- it produces no publishable number.
"""
import argparse
import atexit
import os
import sys

SIZE_KEYS = ("nx", "ny", "nz", "nt", "n_train", "n_val", "n_test",
             "ss_epochs", "ar_epochs", "epochs")

TINY = {"nt": 3, "n_train": 6, "n_val": 3, "n_test": 3,
        "ss_epochs": 2, "ar_epochs": 2, "epochs": 2}

# Marker written into the top level of a result file produced under --tiny. A
# tiny run writes the same filename and the same per-arm schema as the shipped
# archive, so without a marker the two cannot be told apart by content alone.
# The marker travels with the file, which documentation cannot.
TINY_PRESET = "tiny-cpu-pipeline-self-test: not a manuscript value"


def stamp_preset(results, cfg):
    """Tag a top-level result dict with the preset it came from.

    A no-op unless the run used ``--tiny``, so the shipped archive and any
    full-protocol run are left exactly as they were.
    """
    if cfg.get("preset") == "tiny":
        results["_preset"] = TINY_PRESET
    return results
# Note: --tiny does not change the grid size. The number of Fourier modes kept by the
# spectral convolution is built from the grid size: dropping 32x32 to 8x8 makes
# modes=16 > nx//2+1=5 and raises an error outright, so the caller must pass the grid explicitly (--nx/--ny/--nz) and satisfy nx >= 2*modes.


_SKIP_DIRS = {".git", "__pycache__"}


def _snapshot_tree(root):
    """Package-relative POSIX paths of every file under ``root``."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            seen.add(os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/"))
    return seen


def _is_inside(path, root):
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))
    return path == root or path.startswith(root + os.sep)


def _register_repo_watch(repo_root, out_dir):
    """Report at exit every file that appeared inside the package during the run.

    A run whose output root is inside the package is expected to add files there
    and they are listed for the record. A run pointed outside the package must
    add none: if files still appear, some entry point resolved an output path
    from the process working directory rather than from the output root, and the
    package is being polluted by a run that was supposed to avoid it.
    """
    before = _snapshot_tree(repo_root)

    def _report():
        new = sorted(_snapshot_tree(repo_root) - before)
        if not new:
            return
        lines = ["[release-cli] this run added %d file(s) inside the package:"
                 % len(new)]
        lines += ["  " + p for p in new[:20]]
        if len(new) > 20:
            lines.append("  ... and %d more" % (len(new) - 20))
        if _is_inside(out_dir, repo_root):
            lines.append("[release-cli] the output root is inside the package, so this is "
                         "expected; use --output-dir <dir> to keep a run out of it.")
        else:
            lines.append("[release-cli] ERROR: the output root is OUTSIDE the package, yet "
                         "files appeared inside it -- an output path was resolved from the "
                         "working directory instead of from the output root.")
        print("\n".join(lines), file=sys.stderr)

    atexit.register(_report)


def apply_cli(default_seeds, default_results, default_ckpt=None,
              device="cuda", description=""):
    """Parse argv and return ``(seeds, results_file, ckpt_dir, cfg)``.

    ``cfg`` holds the resolved ``device`` plus any size/epoch overrides the
    caller asked for; keys the caller does not use are simply ignored.
    """
    ap = argparse.ArgumentParser(description=description or None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seed", type=int, default=None,
                   help="run a single seed instead of the default set")
    g.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run an explicit list of seeds")
    ap.add_argument("--output-dir", default=None,
                    help="write the result file here instead of the shipped results/ directory")
    ap.add_argument("--ckpt-dir", default=None,
                    help="checkpoint directory (default: <repo>/checkpoints)")
    ap.add_argument("--device", default=None,
                    help="torch device string (default: %s)" % device)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing file in the shipped results/ directory")
    ap.add_argument("--tiny", action="store_true",
                    help="CPU pipeline self-test preset: tiny data, 2 epochs, cpu device. "
                         "Produces no publishable number; use --output-dir to keep it off the archive.")
    for name in SIZE_KEYS:
        ap.add_argument("--" + name.replace("_", "-"), type=int, default=None,
                        help="override the %s used by this run" % name)
    a = ap.parse_args()

    seeds = [a.seed] if a.seed is not None else (list(a.seeds) if a.seeds else list(default_seeds))
    out_dir = a.output_dir or os.path.dirname(default_results)
    results_file = os.path.join(out_dir, os.path.basename(default_results))

    # ``__file__`` is this module, which sits at the repository root, so one
    # dirname -- not two -- is the root.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if a.ckpt_dir:
        ckpt_dir = a.ckpt_dir
    elif a.output_dir:
        # --output-dir moves the CLI-managed results file and checkpoint
        # directory; a run aimed off the package must not be able to leave a
        # CLI-managed checkpoint inside it.
        ckpt_dir = os.path.join(out_dir, "checkpoints")
    else:
        ckpt_dir = default_ckpt or os.path.join(repo_root, "checkpoints")

    shipped = os.path.normcase(os.path.abspath(os.path.dirname(default_results)))
    if (os.path.normcase(os.path.abspath(out_dir)) == shipped
            and os.path.exists(results_file) and not a.force):
        raise SystemExit(
            "REFUSING to overwrite the archived result file:\n  %s\n"
            "It ships with this release and is the record behind the reported values.\n"
            "Pass --output-dir <dir> to write elsewhere, or --force to overwrite it."
            % os.path.abspath(results_file))
    os.makedirs(out_dir, exist_ok=True)
    # The checkpoint directory is deliberately *not* created here: a caller that
    # trains checkpoints creates it itself under the same output root, and a
    # caller that trains none (3d/run_3d_multi_seed.py) must not leave an empty
    # directory behind in the package.

    _register_repo_watch(repo_root, out_dir)

    cfg = {}
    if a.tiny:
        cfg.update(TINY)
        cfg["preset"] = "tiny"
        print(
            "[release-cli] --tiny PRESET ACTIVE: CPU pipeline self-test "
            "(tiny data, 3 time steps, 2 epochs).\n"
            "[release-cli] It exercises the run path end to end and produces NO "
            "manuscript value; the result file is stamped \"_preset\".\n"
            "[release-cli] Pass --output-dir <dir> to keep the run off the archive.",
            flush=True)
    for name in SIZE_KEYS:
        v = getattr(a, name)
        if v is not None:
            cfg[name] = v
    cfg["device"] = a.device or ("cpu" if a.tiny else device)
    return seeds, results_file, ckpt_dir, cfg
