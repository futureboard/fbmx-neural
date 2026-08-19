# FBMX — Futureboard neural DSP

Research infrastructure for training **causal, stateful neural models of audio
effects** and exporting them to `.fbmx`, a self-describing model container meant
to be loaded by a pure Rust realtime runtime.

> **Status: research prototype.** Everything here runs end to end. Two
> teachers exist: a procedurally generated smoke test whose teacher is a toy
> (a one-pole envelope driving a gain, then a tanh), and the **FA76 circuit
> core**, which renders real dry/wet pairs plus gain-reduction and
> control-voltage traces (`crates/fa76-dataset-gen`). A model distilled from
> the circuit inherits that circuit model's approximations; it is not a
> measurement of hardware and must never be described as one. No model here is
> a **validated production model** — `.fbmx` files carry a `validated` flag
> that stays false until a human has measured one against its reference.

Build the laboratory before collecting the specimens.

---

## 1. Research goal

Model audio effects — starting with dynamics — well enough to run inside a
realtime audio callback, with the parameters exposed as real controls rather
than baked into the weights.

Three constraints follow from "realtime plugin", and they shape every decision
below:

| Constraint | Consequence |
|---|---|
| The host gives you a block of unknown size | Block-size-independent output is mandatory, not nice to have |
| The host gives you no future samples | Causal only. No bidirectional layers, no lookahead, no file-level normalisation |
| The callback has a hard deadline | Small models, O(1) per sample, no Python anywhere in the audio path |

The long-term shape:

```
training                model              production inference
Python + PyTorch   ->   .fbmx        ->    pure Rust FBMX runtime
```

## 2. Architecture

```
audio sample  ──┐
                ├─→ [ concat / FiLM ] ─→ LSTM(32) ─→ head ─→ + ─→ output
continuous ─────┤                                            ↑
controls        │                                            │
                │                                        dry signal
categorical ────┘                                       (residual path)
states → embedding
```

The V0 baseline is **LSTM-32**: one layer, 32 hidden units, mono, causal,
48 kHz reference rate, residual output head. **5,289 parameters** with the
smoke-test control set (three parameters → a 6-dimensional conditioning
vector); 4,513 unconditioned.

* `fbmx/models/base.py` — the contract every architecture implements:
  `init_state`, `forward(x, params, state) -> (y, new_state)`, `export_spec`.
* `fbmx/models/rnn.py` — conditioning, output head, residual path, shared by
  LSTM and GRU.
* `fbmx/models/lstm.py`, `gru.py` — the two recurrent variants.
* `fbmx/models/tcn.py` — causal dilated TCN with per-block FiLM and an explicit
  input cache per block. Experimental, untuned.

**Conditioning is generic.** A model is handed a `ConditioningSchema` and
builds whatever projections it implies. Continuous controls (Input, Attack,
Release, Ratio, Drive, Mix, …) are normalised to `[-1, 1]`. Categorical states
(Revision, "all buttons in") get **learned embeddings** — encoding them as a
number on a continuous axis would assert that revision D sits halfway between C
and E, which is false and costs accuracy exactly at the settings people care
about. Nothing in `fbmx/` knows what an FA76 is; the parameter set comes from
the dataset.

Two conditioning mechanisms are implemented: `concat` (appended to the
per-sample input, as in the parametric-RNN line of work) and `film` (per-channel
scale/shift on the recurrent output). `both` is available for experiments.

**Adding S4 / state-space models later requires no trainer change.** Their
recurrent state is just another tensor tree; implement `StreamingModel` and
register it.

## 3. Why causal and stateful is not negotiable

A neural effect eventually lives inside `process(buffer)`. If any temporal
state resets at a block boundary, the output depends on the host's buffer size
— the user hears a click or a pumping artefact every 128 samples, and the model
sounds different after they change their audio device. This is not a quality
issue that more training fixes; it is a correctness bug.

So: state is always explicit and always the caller's to carry
(`fbmx/streaming/inference.py`), and equality between whole-sequence and
block-wise processing is a **test**, at 16/32/64/128/256/512/1024 samples and at
randomly varying block sizes (`tests/test_streaming_equivalence.py`). For
single-layer LSTM in FP32 the difference is exactly zero; for GRU and TCN it is
float32 accumulation noise around 1e-7.

Recurrence also buys unbounded memory at O(1) cost per sample. A compressor's
release can run for hundreds of milliseconds; a convolutional model needs a
receptive field to match, and pays for it in every callback.

## 4. Dataset abstraction

There is **no real dataset here, and nothing in this package downloads one.**
Acquiring audio, and checking that its licence permits what you intend to do
with the resulting model, is a human step on purpose.

Every adapter yields the same thing — dry/wet sequence pairs plus the parameter
setting that produced them — and every adapter must supply a `DatasetInfo`:

```
name · source · source_type · license · license_url · version · sample_rate
attribution · citation · checksum · redistributable · notes
```

`source_type` is one of `synthetic`, `circuit_model`, `hardware_capture`,
`hybrid`. `redistributable` defaults to **False**: a dataset is assumed
un-redistributable until somebody has read its terms. **A dataset being
downloadable does not make it redistributable, and a model inherits the licence
of the data it was fitted to** — which is why the exporter copies these fields
into the `.fbmx` header, and why the export cannot claim a cleaner provenance
than the training run had.

Real corpora arrive through `PairedAudioDataset` plus a **manifest** — a JSON
file listing which files pair with which, at which parameter setting, in which
split, with SHA-256 checksums. The manifest is committed; the audio is not.
That is what lets a hardware capture session, a licensed corpus and a teacher
render all be described in the repository without redistributing a byte of
anyone's audio.

Deliberate non-features in the audio adapter: no downloading, no resampling
(a rate mismatch is an error — resampling a nonlinear effect's training pair
changes the target), no stereo folding.

## 5. The synthetic smoke test, and its limits

`fbmx/datasets/synthetic.py` generates deterministic dry/wet pairs from sines,
multitones, chirps, noise, impulses, transient bursts, amplitude steps and
silence transitions — chosen to cover attack, release, state decay and
broadband behaviour, i.e. the cases where stateful models break.

The teacher is:

```
env[n] = a * env[n-1] + (1-a) * |x[n]|      (attack 5 ms / release 80 ms)
gain   = 1 / (1 + k * env)
y      = mix * shape(pre * x * gain) + (1-mix) * x
```

Nothing about it is derived from any circuit and no parameter has a physical
unit. **Its only job is to have memory**, so that a stateless model cannot fit
it and the streaming tests test something real. Fitting it demonstrates
plumbing: dataset → dataloader → training → loss decreases → checkpoint →
resume → export → streaming inference. **Do not draw any acoustic conclusion
from a model trained on it.**

### The FA76 circuit teacher

Two corpora exist. `fa76-revd-v1` is the phase-2 corpus; `fa76-revd-v2` is the
one to use, and differs in three ways that turned out to matter more than any
model change:

| | v1 | v2 |
|---|---|---|
| dry/wet alignment | 4 whole samples removed, **0.139 left in** | fractional delay removed too, **residual 1e-5 samples** |
| sampling | by dial position | by **gain reduction**, via a calibration sweep of the circuit |
| all-buttons | sampled independently of 20:1 | **matched pairs**: same dry audio, same dials, both ratios |
| sequences | 1000 x 1.0 s | 500 x 2.5 s |

The alignment fix is a 127-tap Kaiser windowed-sinc fractional delay applied to
the wet signal. It is a delay and only a delay: the magnitude response is
untouched to under 0.01 dB across the band (asserted in
`crates/fa76-dataset-gen/src/align.rs`), because correcting a phase error with
an EQ would hide the problem rather than remove it. What is left in the target
afterwards is the circuit's *own* phase response — preamp and transformer
bandwidth — which is part of the effect and must stay.

The first real target is the FA76 circuit core, rendered by
`crates/fa76-dataset-gen`:

```bash
cargo run -p fa76-dataset-gen --release --     --out datasets/fa76-revd-v1 --sequences 1000 --seconds 1.0 --seed 1
cd neural && python scripts/train.py --config configs/fa76_revd.yaml
```

Conditioning: **Input**, **Attack**, **Release** as continuous dials in
their hardware ranges, and **Ratio** as a categorical with five states —
4:1, 8:1, 12:1, 20:1 and *All Buttons*. All Buttons is a different circuit
state, not a larger ratio, so it gets its own embedding row rather than a
position on a ratio axis. OUTPUT is deliberately not conditioned: it sits after
the sidechain tap, cannot change gain reduction, and stays deterministic DSP
outside the model. Revision is fixed to Rev D.

The config declares what it expects of the manifest:

```yaml
data:
  expect: {source_type: circuit_model, teacher: FA76 Circuit, revision: Rev D}
```

Pointing it at a hardware capture, or a different revision, is then a loud
failure rather than a model with a provenance claim nobody checked.

Two facts about this dataset that shape the numbers: the wet/dry level ratio
spans roughly -4 dB to +23 dB across the corpus, because the INPUT dial moves
the whole chain's gain and not just its threshold; and 4 of the circuit's
4.148 samples of oversampling group delay are removed by the renderer, leaving
a 0.148-sample fractional delay the model has to absorb. Both are recorded in
the manifest's `known_mismatches`.

## 6. Training

```bash
python scripts/train.py --config configs/smoke.yaml
python scripts/train.py --config configs/lstm32.yaml
python scripts/train.py --config configs/smoke.yaml --set train.lr=0.001 train.epochs=50
```

Device selection is automatic (CUDA → MPS → CPU); nothing is tuned for a
particular GPU and everything runs on CPU and in Colab. FP32 is the reference;
AMP is opt-in (`train.amp: true`) and ignored where unsupported.

### Training on a GPU

Worth it even for a model this small: measured on a GTX 1060 3 GB against six
CPU cores, at the same batch size and step count, **473 ms → 106 ms per TBPTT
window, ~150 s → ~33 s per epoch**. `scripts/bench_train.py` runs that
comparison on any machine before you commit to it — a 32-unit LSTM is a poor
fit for a GPU on paper, so measure rather than assume.

Two practical notes. The stock PyTorch wheel is CPU-only, and a CUDA build has
to be installed from the PyTorch index; keep it in a *separate* environment so
the CPU one still backs the tests and the golden-vector export:

```bash
python -m virtualenv .venv-cuda
.venv-cuda/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv-cuda/Scripts/python -m pip install numpy pyyaml        # not on that index
.venv-cuda/Scripts/python -c "import torch; print(torch.cuda.get_arch_list())"
```

That last line is the one that matters on older hardware: recent CUDA builds
have been dropping Pascal, and a wheel without `sm_61` will load and then fail
at the first kernel. The cu126 build still carries `sm_50` through `sm_90`.

Leave `amp: false` on Pascal — GP106 runs fp16 at 1/64 rate, so AMP is slower
there, not faster. And note that cuDNN's LSTM backward is not deterministic:
the seed still fixes the data order and the initialisation, but two GPU runs
will not be bit-identical to each other or to a CPU run.

**Truncated BPTT.** The recurrent state is carried across chunk boundaries and
only the autograd graph is cut:

```
state = zeros
for chunk in sequence:                  # contiguous, in order
    y, state = model(chunk, params, state)
    window_loss += loss(y, wet_chunk)
    if window is full:                  # every `tbptt_chunks` chunks
        window_loss.backward()
        clip; optimiser.step(); zero_grad
        state = detach(state)           # state survives, graph does not
```

That is what lets a 32-unit LSTM learn an 80 ms release from 4096-sample chunks.
The first `warmup_samples` of each sequence are excluded from the loss: the
model starts from a zero state the target does not share, and scoring that
transient teaches it to guess.

Defaults: 48 kHz, chunk 4096, hidden 32, float32, small batch.

**Losses** (`fbmx/losses/`) are registry-driven and composable:

```yaml
loss:
  - {name: mae,    weight: 1.0}
  - {name: mrstft, weight: 0.5}
  - {name: dc,     weight: 0.1}
```

Waveform L1 plus multi-resolution STFT is the starting point. Extension points
that will matter for dynamics are already wired: `envelope` and `transient`
losses (no extra targets needed), and `aux_trace` for gain-reduction /
control-voltage supervision when the teacher or rig exports the trace — the
smoke dataset already carries a `gain` trace so that path is exercised. A loss
that cannot find its target raises rather than silently scoring zero.

### Auxiliary dynamics heads

A compressor can be fitted waveform-first and still get its *dynamics* wrong:
the gain trajectory is a small fraction of the signal energy, so it is nearly
free to lose under an L1 loss. When the teacher can export the trajectory —
a circuit model can, a hardware capture usually cannot — the model can be asked
to predict it from the same recurrent state:

```yaml
model:
  aux_heads: [gain]         # audio stays mandatory; this is optional
loss:
  - {name: aux_trace, as: gr, key: gain, weight: 0.5, scale: 0.04}
```

`weight` is λ and `scale` brings a trace in physical units into the range a
freshly initialised head can reach. Both live in the config, never in code. The
heads cost the audio path nothing at inference: their weights are in the `.fbmx`
and the Rust runtime never evaluates them.

Two traces are wired up, and `configs/fa76_revd_v2.yaml` uses both:

```yaml
model:
  aux_heads: [gain, cv]
loss:
  - {name: aux_trace, as: gr, key: gain, weight: 0.5,  scale: 0.04}  # dB -> dB/25
  - {name: aux_trace, as: cv, key: cv,   weight: 0.25, scale: 0.5}   # volts
```

A word of caution from the phase-3 measurements: supervising both traces
improved gain-reduction tracking (1.5 dB MAE) but coincided with a ~1 dB
regression in high-frequency response on a 32-unit state. Whether that is the
heads competing for capacity or the corpus shift is not yet separated — it
needs an ablation, not an opinion.

`scripts/inspect_dynamics.py` measures what the trained state actually holds:
per-unit cell-decay time constants, the implied forget gates, and the release
`t63` the model produces as the Release dial moves. That is how phase 3
established that the release error is *not* a memory-capacity limit.

## 7. Checkpoint and resume

A killed Colab runtime must never cost a run. Every save records weights,
optimiser state, scheduler state, epoch/step counters, best metric, all RNG
streams, the metric history and a snapshot of the config.

```bash
python scripts/train.py --config configs/smoke.yaml --resume checkpoints/smoke/last.pt
```

`last.pt` is written every epoch; `best.pt` when the monitored metric improves.
Resume rebuilds the **architecture from the checkpoint, not from the config**,
so editing a config between runs cannot silently change the model underneath a
resume. Checkpoints load with `torch.load(weights_only=True)`.

Checkpoints are development artifacts tied to this PyTorch version. They are
not the distribution format.

## 8. FBMX export

```bash
python scripts/export_fbmx.py --checkpoint checkpoints/smoke/best.pt --output models/smoke.fbmx
python scripts/inspect_fbmx.py models/smoke.fbmx --tensors --load
```

`.fbmx` v1 layout:

```
0   4   magic b"FBMX"
4   4   u32 LE format version
8   8   u64 LE header length
16  H   UTF-8 JSON header
    pad zeros to a 16-byte boundary
D   ..  tensor data, little-endian, C-contiguous, in header order
-32 32  sha256 of every preceding byte
```

The header carries: format version, model UUID, creation time, producer,
model type/architecture, sample rate, channels, causal/recurrent flags,
receptive field, parameter count, hidden size, full hyper-parameters, input
feature spec, **state spec**, conditioning schema, normalisation block, the
tensor table (name/dtype/shape/offset/nbytes), metadata (name, author, licence,
attribution, `model_source_type`, dataset provenance, training summary, tags,
`validated`), and checksums for the tensor region.

Rules: **no pickle** (loading a `.fbmx` cannot execute anything — there is a
test asserting the bytes contain no pickle opcodes); tensor data starts on a
16-byte boundary so a Rust reader can cast slices; both checksums are verified
on read; every architecture round-trips bit-exactly and streams identically
after reload.

**ONNX** (`fbmx/export/onnx_ref.py`) exists only as a second opinion while the
Rust runtime is written. Neither `onnx` nor `onnxruntime` is a dependency and
nothing imports it by default.

## 9. Streaming inference

```python
from fbmx.export import read_fbmx
from fbmx.streaming import StreamingProcessor

model = read_fbmx("models/smoke.fbmx").build_model()
proc = StreamingProcessor(model, model.schema.encode({"drive": 0.7, "mode": "hard"}))
proc.reset()                       # on transport stop
out = proc.process(block)          # any block size, state carried
```

`latency_samples` is 0 by construction. Parameters are read once per block;
per-sample smoothing of the conditioning vector is a runtime concern and is
deliberately not simulated here. `streaming_equivalence()` reports the max
absolute difference between whole-sequence and blocked processing — that is the
function the tests call, and the number to check first when something sounds
wrong.

## 10. Running the model from Rust

`.fbmx` exists to be loaded by something that is not Python.
`crates/fbmx-runtime` is that something: a pure Rust loader and realtime LSTM
kernel with no Python, no PyTorch, no ONNX Runtime and no allocation after
`load()` returns. See `docs/fbmx-format-v1.md` for the format and
`crates/fbmx-runtime/src/lib.rs` for the API.

The two implementations are held together by **golden vectors**:

```bash
python scripts/make_golden.py --fbmx models/fa76-revd.fbmx     --out tests/golden/fa76_revd --set Input=8 Attack=6 Release=4 Ratio="All Buttons"
cargo test -p fbmx-runtime --test golden -- --nocapture
```

`make_golden.py` writes the probe signal, the conditioning values, PyTorch's
output and its final `h`/`c` next to a copy of the model. The Rust test loads
the same file, processes the same signal from a zero state, and must match
within 1e-6 on both audio and state. The fixtures live in `tests/golden/` and
are committed — they are the contract between the two halves of the project,
and regenerating them is a deliberate act.

## 11. Research references

Used as references, not reproduced. Where this implementation differs, it says
so in the module docstring.

| Paper | What we took | Where we differ |
|---|---|---|
| **NablAFx** — A Framework for Differentiable Black-box and Gray-box Modeling of Audio Effects, arXiv:2502.11668 | Separating backbone / conditioning / loss into independently swappable pieces | We fix the streaming contract at the model base class; gray-box differentiable DSP blocks are not implemented yet |
| **Differentiable Black-box and Gray-box Modeling of Nonlinear Audio Effects**, arXiv:2502.14405 | Conditioning as a first-class object rather than extra input channels | Same |
| **Efficient Neural Networks for Real-time Modeling of Analog Dynamic Range Compression**, arXiv:2102.06200 | Causal dilated TCN + FiLM; L1 + multi-resolution STFT loss | Their evaluation is offline on whole segments; ours must be exact under block processing, so every conv block carries an explicit input cache. No batch/global normalisation anywhere |
| **Condition Mechanisms for Black-box Audio Effect Modeling**, arXiv:2408.04829 | FiLM-family conditioning generally beats plain concatenation | We default to concatenation for the realtime baseline (cheapest per sample) and expose FiLM as an experiment rather than a claim. Time-varying variants (TFiLM/TVFiLM) pool over a window, which makes the result block-size dependent — excluded from the realtime baseline |
| **Modeling Analog Dynamic Range Compressors using State Space Models**, arXiv:2403.16331 | S4/SSM as the eventual long-memory backbone | Not implemented. The model contract is shaped so it can be added without touching the trainer |
| **PANAMA — Parametric Neural Amp Modeling with Active Learning**, arXiv:2509.26564 | Parametric recurrent modelling; active selection of parameter points during capture | Active learning is a capture-time strategy and belongs with the (not yet existing) capture tooling; the dataset manifest is designed to record which points were sampled |

Local copies of some of these are in `../references/arxiv/`.

---

## Layout

```
neural/
├── fbmx/
│   ├── conditioning.py     ConditioningSchema / ParamBatch — the control-surface contract
│   ├── config.py           YAML configs, `extends`, dotted overrides
│   ├── device.py           auto device selection
│   ├── datasets/           base (provenance) · manifest · synthetic · paired_audio
│   ├── models/             base · rnn · lstm · gru · tcn · registry
│   ├── losses/             waveform · spectral · aux (extension points) · registry
│   ├── training/           trainer (TBPTT) · checkpoint · metrics
│   ├── export/             fbmx (the container) · onnx_ref (optional)
│   └── streaming/          inference (StreamingProcessor, equivalence checks)
├── configs/                smoke.yaml · lstm32.yaml · fa76_revd.yaml
├── scripts/                train · validate · export_fbmx · inspect_fbmx · make_golden
└── tests/                  incl. golden/ — fixtures shared with the Rust runtime
```

## Install and test

```bash
pip install -r requirements.txt        # torch, numpy, pyyaml, pytest
python -m pytest tests -q              # ~30 s on CPU, incl. the full CLI pipeline
```

CUDA tests skip themselves when no GPU is present. For CUDA wheels, install
torch from the PyTorch index first.

## What this is not, yet

* Not fitted to any hardware, and not validated against any physical unit.
* No hardware capture tooling, no active learning (the dataset sampler is
  structured for it; it is not implemented).
* No gray-box / differentiable-DSP blocks, no S4 backbone.
* No quantisation, and `.fbmx` stores float32 only.
* Parameters are constant within a training sequence; automation is not
  modelled yet.
