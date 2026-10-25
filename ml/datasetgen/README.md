# Hole-fill training-data harness (seeded, engine-driven)

Generates short-sequence samples for the 6ch hole-fill CNN. Every sample is
rendered by the **real engine** and warped by the **real VulkanWarp pull
pass** — no synthetic scenes, no numpy warp reimplementation.

Atomic sample = one **sequence**: a source frame at `P0`, true frames at
`P1_{t-1}` and `P1_t`. `input_t` is the real pull-pass `warp(P0 -> P1_t)`:
warped RGB + splat depth + hole mask, plus `prev_t` = the previous **true**
frame reprojected by the small delta (teacher forcing, standing in for the
previous filled frame). Output supervision is the true frame, hard-masked at
composite time.

## Files

| file | role |
|------|------|
| `capture.cpp` | standalone C++ capture binary (GL scene render + VulkanWarp). Compiled by hand, no CMake edits |
| `capture` | built binary (not committed) |
| `gen_dataset.py` | seeded driver: pose sampling, direction coupling, gating, packaging, manifests |
| `verify_sample.py` | blur-baseline MAE check on one packaged sample |
| `../data.py` | PyTorch `HoleFillDataset` → `(input6ch, target_hole_rgb, mask)` at 512x512 |

## Build

From `openwarp/`, no CMake changes (links the existing
`build/libopenwarp_vulkan.a` plus already-built vendored GL libs):

```
g++ -std=c++17 -O2 -DGLEW_NO_GLU -DGLEW_STATIC \
  -Iinclude/eigen -Iinclude/glfw/include -Iinclude/glew-cmake/include \
  -Isrc/openwarp ml/datasetgen/capture.cpp src/openwarp/util/obj.cpp \
  build/libopenwarp_vulkan.a build/lib/libglewd.a \
  build/include/glfw/src/libglfw3.a \
  -lvulkan -lGL -lX11 -lXext -ldl -lm -lpthread -o ml/datasetgen/capture
```

Needs a running X server for the hidden GL window (headless CI: `Xvfb :77
-screen 0 1280x1024x24 &`, then `DISPLAY=:77 ...`). The Vulkan warp itself
is fully headless (lavapipe `llvmpipe` works; the smoke set below was
captured on it).

## Regen

Smoke (what CI/a reviewer runs — seconds, ~6 MiB):

```
cd openwarp
DISPLAY=:77 ml/.venv/bin/python ml/datasetgen/gen_dataset.py \
  --out data --seed 0 --train 4 --val 2
```

Full default (1000 train / 200 val):

```
DISPLAY=:77 ml/.venv/bin/python ml/datasetgen/gen_dataset.py \
  --out data --seed 0 --train 1000 --val 200
```

Same `--seed` + same flags regenerates **byte-identical manifests**
(`manifest.json`, per-seq `__meta.json`: sorted keys, no timestamps) and
numerically identical npz on the same machine/driver. Verified by running
the smoke set twice and `cmp`-ing manifests plus array-comparing every npz.

## Motion model and axis flags

Fixed scene: exactly the repo's demo scene (`resources/scene.obj` +
bakes), anchor = `TestRun` start pose `(-3, 1.5, 2)` / yaw `-45°`. No
lighting/scene variation in this first cut.

Per-axis tunable ranges (each `--flag x y z`, half-extents; rotations in
degrees, euler `dq = qx·qy·qz` right-multiplied = camera-local delta):

| group | translation (m) | rotation (deg) | default |
|-------|-----------------|----------------|---------|
| anchor jitter around scene anchor | `--anchor-pos` | `--anchor-rot` | `0.4 0.25 0.4` / `4 4 4` |
| warp delta `P0->P1` (large, makes holes) | `--warp-pos` | `--warp-rot` | `0.10 0.06 0.10` / `1.5 1.5 1.5` |
| inter delta `P1_{t-1}->P1_t` (small, ~90 Hz step) | `--inter-pos` | `--inter-rot` | `0.012 0.008 0.012` / `0.2 0.2 0.2` |

Same-direction coupling (never opposing): the inter delta is sampled in its
own ranges, sign-aligned per axis to the warp delta, then gated on cosine
similarity (`--min-cos`, default 0.5, translation and rotation separately)
and on magnitude ordering (`|inter| <= |warp|`, translation norms and
rotation norms). Violations resample (counted in `manifest.stats[].rejected`
as `cos-trans`/`cos-rot`/`mag-trans`/`mag-rot`). Typical accepted cosine
similarities on the smoke set: 0.6–0.96 (same direction with variance).

Hole gate: `--hole-min/--hole-max` (default 0.02–0.15) on the full-res warp;
rejects resample (counted as `hole-gate`). `--max-factor` (default 10)
bounds total candidates per split; exceeding it aborts loudly. `--batch`
(default 32) is shots per `capture` subprocess call (one GL context + one
VulkanWarp init amortized over the batch).

## Output layout and costs

```
data/
  manifest.json                 # seeds, flags, anchor, git commit, per-seq metas, stats
  train/seq000000__src.npz      # {rgb: uint8 512x512x3}
  train/seq000000__true_tm1.npz # {rgb}
  train/seq000000__true_t.npz   # {rgb}
  train/seq000000__input_t.npz  # {warped_rgb u8, depth_u16 u16, mask u8 0/1, prev u8 luma}
  train/seq000000__meta.json    # poses, deltas, seq_seed, hole fracs, cos sims
  val/...
```

Measured on llvmpipe (ARM CI host): **~0.44 s/seq, ~1 MiB/seq** packaged.
Full default ≈ 9 min wall, ≈ 1.2 GiB. Raw 1024 px bins (~32 MiB/seq) are
deleted per batch unless `--keep-raw`; peak transient ≈ one batch.

Downsampling 1024→512: plain 2x2 mean for true/src RGB and prev luma;
**masked** 2x2 mean (valid-pixel average, all-hole → 0) for warped RGB and
splat depth; mask is any-hole-per-block (conservative: every partially-hole
block is supervised). Storage is uint8/uint16; `ml/data.py` converts to the
float32 0..1 6ch net contract (`warped/255`, `depth_u16/65535`, mask,
`prev/255`).

## Checks

```
ml/.venv/bin/python ml/datasetgen/verify_sample.py --data data --split train --index 0
```

Runs the committed `ml/weights/fill_blur_init.onnx` (ORT CPU) on the
sample: asserts valid-pixel output ≈ reference 3x3 box blur (< 0.02, same
bar as `export_blur.py`), hard-composite valid passthrough bit-exact (0),
and reports hole-pixel MAE (informational — the headroom a trained net must
close; ~0.3–0.7 on smoke samples).

`ml/data.py` batch check:

```
ml/.venv/bin/python -c "import sys; sys.path.insert(0,'ml'); import data"
```

## Fidelity notes (read before training on this)

1. **Real warp, NumPy packaging.** Hole RGB/depth/mask come from the actual
   `warp_splat`/`warp_pull`/`warp_fill` shaders via `VulkanWarp`
   (`fillMode 0`, `holeRadius 0`). No warp-math reimplementation exists
   anywhere in this harness — the documented-acceptable numpy fallback was
   not needed.
2. **Splat depth** is decoded straight from the engine's splat plane
   (`enc/16777215`, `0xffffffff` = empty → 0), i.e. target-view nonlinear
   depth. No estimation involved.
3. **Hole mask** = splat-empty OR pull-output alpha < 0.5. The alpha term
   catches pull-stage rejects (back-projection behind the source camera or
   out of bounds, typically frame borders, ~1e-5 of pixels on smoke data);
   the spec's "from splat alpha" is this validity signal. Both fractions
   are recorded separately in every `__meta.json`.
4. **Teacher-forced prev**: `prev_t = warp(true_{t-1} -> P1_t)` luma through
   the same pull pass (holes → 0). At inference the net will see its *own*
   previous output instead — a train/inference skew to revisit (exposure
   bias); `HoleFillDataset(prev_mode='zero')` gives the no-history endpoint.
5. **Render parity**: camera matrix, `perspective(45,1,0.1,100)`, clear
   color, cull/depth state, FBO formats and readback types are copied from
   `OpenwarpApplication::{initGL,renderScene,doVulkanMeshWarp}`. Context
   requests 3.0 like the app (Mesa grants 4.5-compat, which compiles the
   `#version 450` demo shaders). GL renderer + Vulkan device strings land
   in `manifest.json` for provenance.
6. **Cross-machine determinism**: RNG/pose sampling is platform-independent,
   but rendered/warped pixels depend on GPU/driver (llvmpipe vs NVIDIA can
   differ in last-ulp filtering), so `hole_frac` and npz bytes can wobble
   across machines near a gate boundary. Byte-identical regen is guaranteed
   on the same software/driver stack.
7. **Warped-valid ≈ true, not ==**: the pull pass resamples with linear
   filtering, so warped valid pixels differ from a direct render at the
   target pose by ~1 LSB (`valid MAE ≈ 1/255` measured). Supervision on
   holes uses the direct render — unaffected.
