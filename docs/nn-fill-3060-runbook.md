# NN hole-fill perf runbook — 3060 host (mode 1: `VulkanFillMethod::NnFill`)

Why mode 1 ran at ~8fps is measurable on the host now: `VulkanNnFill::Fill`
reports per-stage ms (scan / build-input / infer / composite), the exact hole
fraction, and the active backend on every 1 s rolling summary, and the app
paces the same split every ~2 s in its `[vulkan] stats` line. Start from the
logs, not from guesses.

## 0. One-line symptom map (measured, ARM lavapipe, 1024x1024, ~22% holes)

| build | scan | build | infer (mock blur) | comp | Fill total |
|-------|------|-------|-------------------|------|------------|
| `-O0` Debug | 0.7 | 8.2 | 30.5 | 15.9 | **55.3 ms** |
| `-O3` Release | 0.3 | 2.2 | 1.6 | 2.2 | **6.3 ms** |

Debug alone eats ~55 ms of a 125 ms frame — and that is the *new* code. The
old composite did 3 full-res upsample + 3 full-res scan passes (vs 1 fused
hole-only pass now) plus 3 big mallocs per warp, so old-Debug was worse
still. On the 3060 the same ratios apply; the fix order below follows them.

## 1. Confirm which backend is actually active

Build, then run mode 1 (press `T` until "NN fill", or start warped) with:

```
OPENWARP_NN_LOG=1 ./build/openwarp --vulkan
```

`OPENWARP_NN_LOG=1` = one line per warp; default `0` = 1 s rolling summary
only. Look for the state-change lines (printed once at init/fallback):

```
[nn-fill] mock backend (no ORT C++ at build time): ... (providers=mock ...)   # CPU box blur, NOT the net
[nn-fill] ORT backend: ... (providers=ORT[cpu] threads=8 ...)                 # real net, CPU EP
[nn-fill] ORT backend: ... (providers=ORT[cuda,cpu] threads=8 ...)            # real net, CUDA EP  <- want this
[nn-fill] CUDA EP unavailable (...); using CPU EP.                            # GPU ORT missing/broken -> CPU
[nn-fill] WARNING: unoptimized Debug build (NDEBUG off): ...                 # you are measuring -O0, stop
[vulkan] nn-fill ready: backend=ORT[cuda,cpu] weights=ml/weights/fill_blur_init.onnx
```

If you see `providers=mock`, the C++ never linked ORT (§3). If you see the
Debug WARNING, rebuild Release first (§2) — all other numbers are meaningless.

Per-interval app line (`[vulkan] stats`, every ~2 s) now ends with:

```
... | nn fills=120 early=40 hole=3.1% fill=4.2 (scan=0.1 build=1.1 infer=2.3 comp=0.7) backend=ORT[cuda,cpu]
```

`cpu-warp` in the same line *includes* the fill; subtract `fill` for the pure
GPU-warp + splat-download part.

## 2. Reconfigure Release (the old CMake forced Debug)

`set(CMAKE_BUILD_TYPE Debug)` used to shadow the cache, so even
`-DCMAKE_BUILD_TYPE=Release` built `-O0`. Fixed: unset type now defaults to
Release. On the 3060:

```
git pull                      # pick up this fix
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target openwarp_vulkan_shaders openwarp-vulkan-test openwarp
```

Verify the flags actually took (must show `-O`, not bare `-g`):

```
grep CXX_FLAGS build/CMakeFiles/openwarp_vulkan.dir/flags.make
# want: CXX_FLAGS = -O3 -DNDEBUG -std=gnu++17
```

Stale objects from a pre-fix build dir keep old flags — if in doubt, wipe
`build/` and configure fresh.

## 3. Point at the ORT C++ tarball (else you benchmark the mock)

Python `onnxruntime` wheels ship no C++ headers/libs, so `OPENWARP_HAS_ORT`
stays off until you unpack a release tarball, e.g.:

```
# CPU first (proves the backend switch + real-net infer cost):
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz -C /opt
cmake -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.22.0
# GPU (CUDA EP): use the -gpu tarball + a matching CUDA/cuDNN, same -D flag.
cmake --build build --target openwarp_vulkan_shaders openwarp-vulkan-test openwarp
```

Configure must print `onnxruntime C++ found: ... - NN fill uses ORT`
(not `... not found - NN fill uses mock box-blur backend`). At runtime the
`providers=` field (§1) is the ground truth: `ORT[cuda,cpu]` > `ORT[cpu]` >
`mock`. Tuning knobs: `OPENWARP_NN_THREADS=N` (intra-op pool, default =
core count), `OPENWARP_NN_CUDA_DEVICE=0`.

## 4. Read the summary, map each stage to its budget

| stage (`nn` split) | budget | over budget => |
|--------------------|--------|----------------|
| `scan` (hole pre-scan) | part of pre, <0.1 ms | impossible-ish; indicates throttling |
| `build` (1024² RGBA8+splat → 1x6x512²) | pre <0.5 ms | still Debug, or thread starvation |
| `infer` | Python gate numbers at 512²: inductor fused fp16 **0.78 ms**, ORT-CUDA device-resident **1.85 ms**, ORT-CUDA sync **4.8 ms** | `providers=` wrong (mock/CPU); or per-run H2D/D2H copies (CUDA EP present but feed is CPU — expected for now, bounded by ~4.8 ms) |
| `comp` (bilinear + hard-mask composite) | post <0.5 ms | still Debug, or huge hole fraction (scales with holes by design) |
| every-warp splat download | inside `cpu-warp − fill` | compare vs mode 0 (`cpu-warp` with 1/60 coverage): the delta is PCIe 4 MiB + 1M CPU scan. Required as the hole mask; only suspicious if it exceeds ~2 ms |

Healthy Release target on the 3060: `fill` ≈ pre 0.3 + infer ~1–2 (CUDA)
+ comp ~0.2 ≈ **under ~3 ms/warp**, `early=` high whenever the camera is
still (hole-free frames skip infer entirely).

## 5. What was NOT changed

- `vulkan_warp.cpp` is zero-diff: mode 0 (Off/black) is bit-identical.
- `ml/`, `resources/`, `mise.toml`, test files untouched.
- Fallback safety kept: any load/infer failure latches black (`DisableToBlack`),
  never garbage; the early-out fires only on exactly-zero holes (a `~0`
  threshold would leave black specks and order-dependent output).
