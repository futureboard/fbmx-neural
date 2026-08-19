# Datasets

This directory holds **manifests**, not audio.

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
