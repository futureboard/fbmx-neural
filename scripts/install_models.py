"""Install exported `.fbmx` models where the plugin and playground look.

    python scripts/install_models.py --list
    python scripts/install_models.py
    python scripts/install_models.py --dry-run

The destination is the folder `fa76-neural` searches:

    ~/Documents/Futureboard Studio/Utilities/Neural Models/

or whatever `FBMX_MODEL_DIR` says, which is the same rule the Rust side uses.

Only models that an FA76 host can actually run are installed: the container
has to declare the four controls the engine drives (Input, Attack, Release and
a categorical Ratio). The synthetic smoke model is deliberately left behind —
it loads fine and is a different effect, and putting it in the folder would
just give the user a menu entry that fails.

Nothing is deleted. Re-running overwrites a model whose contents changed and
leaves the rest alone, so it is safe to run after every export.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.export.fbmx import read_fbmx

#: Exported model -> the name it is installed under. The installed name is what
#: the plugin shows and what it persists in a project, so it is chosen here
#: rather than inherited from whatever the training run happened to be called.
CATALOGUE: list[tuple[str, str]] = [
    # Installed names carry no version. The plugin persists the file stem as
    # the selection key in the project, so a name with a version in it means
    # every improved model is a *different* model to a saved session: the host
    # reopens, cannot find "FA76 Rev D LSTM-32 v3", and falls back. A stable
    # name updates in place and saved projects keep playing.
    #
    # The names that remain describe character, not vintage: the long-release
    # model is a different-sounding thing, not an older one. The repository
    # keeps its versioned exports under `models/` because those identify which
    # training run produced what, which is the opposite requirement.
    ("models/fa76-revd-v3.fbmx", "FA76 Rev D LSTM-32.fbmx"),
    ("models/fa76-revd-normmae.fbmx", "FA76 Rev D LSTM-32 (long release).fbmx"),
]

#: Controls an FA76 host drives. A model without them is some other effect.
REQUIRED_CONTINUOUS = ("Input", "Attack", "Release")
REQUIRED_CATEGORICAL = "Ratio"


def model_dir() -> Path:
    """Mirror of `fa76_neural::model_dir`."""
    override = os.environ.get("FBMX_MODEL_DIR", "").strip()
    if override:
        return Path(override)
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME") or "."
    return Path(home) / "Documents" / "Futureboard Studio" / "Utilities" / "Neural Models"


def is_fa76_model(path: Path) -> tuple[bool, str]:
    """Can an FA76 host drive this model? Returns (ok, reason)."""
    try:
        container = read_fbmx(path)
    except Exception as exc:  # a corrupt file is a reason, not a crash
        return False, str(exc)
    schema = container.schema
    continuous = {p.name for p in schema.continuous}
    categorical = {p.name for p in schema.categorical}
    missing = [n for n in REQUIRED_CONTINUOUS if n not in continuous]
    if missing:
        return False, f"no {', '.join(missing)} control (has {schema.names})"
    if REQUIRED_CATEGORICAL not in categorical:
        return False, f"no categorical {REQUIRED_CATEGORICAL} (has {sorted(categorical)})"
    return True, container.metadata.name or path.stem


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Install .fbmx models for the FA76 hosts")
    p.add_argument("--dry-run", action="store_true", help="say what would happen, change nothing")
    p.add_argument("--list", action="store_true", help="show what is installed already")
    p.add_argument("--dest", type=Path, default=None, help="override the destination folder")
    args = p.parse_args()

    dest = args.dest or model_dir()
    print(f"models folder  {dest}")

    if args.list:
        existing = sorted(dest.glob("*.fbmx")) if dest.exists() else []
        if not existing:
            print("  (nothing installed)")
            return 0
        for path in existing:
            ok, detail = is_fa76_model(path)
            mark = "ok " if ok else "BAD"
            print(f"  [{mark}] {path.name:<44} {detail}")
        return 0

    here = Path(__file__).resolve().parent.parent
    installed = 0
    skipped = 0
    for source_rel, install_as in CATALOGUE:
        source = here / source_rel
        if not source.exists():
            print(f"  skip    {source_rel} (not exported yet)")
            skipped += 1
            continue
        ok, detail = is_fa76_model(source)
        if not ok:
            print(f"  refuse  {source_rel}: {detail}")
            skipped += 1
            continue

        target = dest / install_as
        if target.exists() and digest(target) == digest(source):
            print(f"  current {install_as}")
            continue

        verb = "would install" if args.dry_run else ("update " if target.exists() else "install")
        print(f"  {verb} {install_as}  <- {source_rel}  ({detail})")
        if not args.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            installed += 1

    if not args.dry_run and installed:
        print(f"\ninstalled {installed} model(s); {skipped} skipped")
        print("The plugin picks these up when the host next prepares it;")
        print("the playground when it next opens an audio device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
