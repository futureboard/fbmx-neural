"""URMP performance-intelligence dataset pipeline.

URMP answers a different question from VSCO. VSCO is the *acoustic* source —
what a violin sounds like playing a given articulation at a given pitch — and
it is what the SFM voicebank is built from. URMP is a corpus of real people
playing written music, and what it can teach is *how a violinist plays a
score*: where they place a note relative to where it was written, how long they
actually hold it, how the pitch moves inside it, when they vibrate and how
fast, and how one note joins the next.

Those two roles stay separate on purpose. Nothing in this package builds or
replaces a voicebank, and nothing here produces audio.
"""

from __future__ import annotations

#: Bumped whenever the meaning of a produced record changes.
PIPELINE_VERSION = 1

DATASET_NAME = "urmp-performance"
DATASET_LICENSE = "CC BY-NC-SA 4.0"
DATASET_ATTRIBUTION = (
    "University of Rochester Multi-Modal Music Performance (URMP) dataset; "
    "Li, Liu, Tang, Chen, Sharma, Duan."
)
