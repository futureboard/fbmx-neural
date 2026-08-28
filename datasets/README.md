# Datasets

This directory holds **manifests**, not audio.

The VSCO 2 CE ingestion pipeline lives at `neural/datasets/vsco` and is run
from the workspace root with:

```bash
python -m neural.datasets.vsco prepare \
  --source solfage-datasets/VSCO-2-CE \
  --dataset-root solfage-datasets/vsco2-ce
```

It writes deterministic JSONL sample manifests and measured metadata under
`solfage-datasets/vsco2-ce/`. The source checkout is never modified or moved.

A manifest (`<name>/manifest.json`) records which dry/wet files pair with each
other, at which parameter setting, in which split, plus SHA-256 checksums and a
`DatasetInfo` provenance block. See `fbmx/datasets/manifest.py` for the schema
and `tests/test_datasets.py` for a worked example.

Audio lives next to the manifest (`<name>/audio/…`) and is git-ignored. Nothing
in this repository downloads it. Before adding a dataset, record honestly:

* where it came from (`source`, `source_type`)
* what its licence permits — including whether models fitted to it may be
  distributed, and under what terms (`license`, `license_url`,
  `redistributable`)
* what attribution is required, verbatim (`attribution`, `citation`)
* which version/checksum you actually used

A dataset being downloadable does not make it redistributable, and a model
inherits the licence of the data it was fitted to.

To create residual-training pairs for the current Solfege physical backend,
build `solfege-render` and run from `neural`:

```powershell
cargo build -p solfege-tools --release --manifest-path ..\solfege-engine\Cargo.toml
python scripts\prepare_vsco_fbmx.py
```

This writes `manifests/violin-fbmx.json` with the generic bowed-string render
as input and downmixed VSCO Solo Violin as target. It is a research baseline,
not a fitted violin model.

Compile the complete self-contained SFM voicebank from all accepted Solo Violin
records with the Rust model compiler:

```powershell
cargo run -p solfege-tools --bin solfage-model --manifest-path ..\solfege-engine\Cargo.toml -- `
  build-voicebank `
  --dataset ..\solfage-datasets\vsco2-ce `
  --physical ..\artifacts\solfage\SoloViolin\physical.json `
  --fbmx ..\artifacts\solfage\SoloViolin\SoloViolinResidual.fbmx `
  --output ..\artifacts\solfage\SoloViolin\SoloViolin.sfm
```

`INDX` stores typed note/articulation/dynamic/round-robin entries and
attack/sustain/release ranges. `AUDO` stores every accepted recording as
canonical interleaved PCM16, while `ACOU` stores only derived descriptors. The
compiler embeds the data into the SFM, so playback does not depend on the VSCO
dataset being present at runtime.
