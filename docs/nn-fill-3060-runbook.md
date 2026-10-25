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
[nn-fill] mock backend (no Torch/ORT C++ at build time): ... (providers=mock ...)   # CPU box blur, NOT the net
[nn-fill] ORT backend: ... (providers=ORT[cpu] threads=8 ...)                 # real net, CPU EP
[nn-fill] ORT backend: ... (providers=ORT[cuda,cpu] threads=8 ...)            # real net, CUDA EP
[nn-fill] Torch backend: ... (device=cuda threads=8 ...)                     # real net, TorchScript CUDA  <- want this
[nn-fill] Torch backend: ... (device=cpu threads=8 ...)                      # real net, TorchScript CPU
[nn-fill] Torch CUDA unavailable (...); using CPU.                            # libtorch CPU-only or no driver -> CPU
[nn-fill] CUDA EP unavailable (...); using CPU EP.                            # GPU ORT missing/broken -> CPU (legacy path)
[nn-fill] WARNING: unoptimized Debug build (NDEBUG off): ...                 # you are measuring -O0, stop
[vulkan] nn-fill ready: backend=TORCH[cuda,cpu] weights=ml/weights/fill_blur_init.pt
```

Backend priority when several are compiled in (decided: torch's CUDA
stacking works on the 3060 host — proven by the eager/inductor runs —
while ORT's prebuilt CUDA EP keeps failing loader-side):

| compiled in | files present | backend used |
|-------------|---------------|--------------|
| Torch (+ORT) | `.pt` (as-given or `.onnx`→swapped twin) | `TORCH[cuda,cpu]` / `TORCH[cpu]` |
| Torch + ORT | only `.onnx` | `ORT[cuda,cpu]` / `ORT[cpu]` |
| ORT only | `.onnx` (as-given or `.pt`→swapped twin) | `ORT[cuda,cpu]` / `ORT[cpu]` |
| any | only the other twin | `mock` (loud line, still needs a file) |
| none | either | `mock` |

`Load` accepts either extension and probes the twin (`fill_blur_init.onnx`
⇄ `fill_blur_init.pt`), so the caller layouts (env `OPENWARP_NN_MODEL`,
build default, relative `ml/weights/` paths) need no changes.

If you see `providers=mock`, neither C++ backend linked (§3/§3c). If you
see the Debug WARNING, rebuild Release first (§2) — all other numbers are
meaningless.

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

(Torch preferred: if you follow §3c below, this section is legacy —
both can stay compiled, Torch wins at runtime.)

Python `onnxruntime` wheels ship no C++ headers/libs, so `OPENWARP_HAS_ORT`
stays off until you unpack a release tarball, e.g.:

```
# CPU first (proves the backend switch + real-net infer cost):
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz -C ~/opt
./build.sh --ort-root ~/opt/onnxruntime-linux-x64-1.22.0
# GPU (CUDA EP): use the gpu_cuda13 tarball (matches the pinned 1.31
# onnxruntime-gpu wheels), same script:
./build.sh --ort-root ~/opt/onnxruntime-linux-x64-gpu_cuda13-1.31.0
```

Configure must print `onnxruntime C++ found: ... - NN fill uses ORT`
(not `... not found - NN fill uses mock box-blur backend`). At runtime the
`providers=` field (§1) is the ground truth: `ORT[cuda,cpu]` > `ORT[cpu]` >
`mock`. Tuning knobs: `OPENWARP_NN_THREADS=N` (intra-op pool, default =
core count), `OPENWARP_NN_CUDA_DEVICE=0`.

### 3b. Hermetic CUDA runtime — no env vars, no sudo (3060)

`./build.sh` stages this automatically (needs `patchelf`:
`sudo apt install patchelf`, one-time): it symlinks the pip-pinned
NVIDIA libs (cublasLt/cublas/cudart/curand `.so.13/.so.10`,
`libcudnn.so.9` + a `libcudnn.so` alias for ORT's lazy bare-soname
`dlopen`) from the mise venv into the ORT tarball `lib/` dir and bakes
`RUNPATH=$ORIGIN` into `libonnxruntime_providers_cuda.so` with
patchelf. The provider lib ships with no RUNPATH, so without this its
DT_NEEDED *and* its lazy `dlopen("libcudnn.so")` / cublas lookups
ignore anything but the system loader search — which is exactly what
broke under gamemode's sanitized environment. After staging, plain
`ldd` is clean and no `LD_LIBRARY_PATH`/`ldconfig` is needed; re-running
`./build.sh` is a no-op when links + RUNPATH are already right (it also
re-stages after `setup-cuda` refreshes the venv — symlinks would dangle
otherwise).

Manual equivalent (what the script does, for debugging):

```
ORTLIB=~/opt/onnxruntime-linux-x64-gpu_cuda13-1.31.0/lib
VENV=~/Desktop/openwarp/ml/.venv/lib/python3.12/site-packages/nvidia
ln -sf $VENV/cublas/lib/libcublasLt.so.13 $VENV/cublas/lib/libcublas.so.13 \
       $VENV/cuda_runtime/lib/libcudart.so.13 $VENV/curand/lib/libcurand.so.10 \
       $VENV/cudnn/lib/libcudnn.so.9 $ORTLIB/
ln -sf $VENV/cudnn/lib/libcudnn.so.9 $ORTLIB/libcudnn.so
patchelf --set-rpath '$ORIGIN' $ORTLIB/libonnxruntime_providers_cuda.so
ldd $ORTLIB/libonnxruntime_providers_cuda.so | grep 'not found'  # must be empty
```

In-app fallback (unchanged): at ORT init the binary still `dlopen()`s
the provider's DT_NEEDED CUDA libs by absolute path (`RTLD_GLOBAL`)
from the ORT lib dir → `LD_LIBRARY_PATH` → venv globs, logged with
`OPENWARP_NN_LOG=1`; any still-missing lib is named loudly if the CUDA
EP then fails. `libcuda.so.1` must come from the NVIDIA driver — no
driver, no CUDA EP, loud CPU fallback.

### 3c. TorchScript backend (preferred — replaces the ORT C++ path)

Decision (do not relitigate): torch's CUDA stacking works reliably on
the 3060 host (proven by the Python eager/inductor runs), while ORT's
prebuilt CUDA EP keeps failing on loader issues. `VulkanNnFill` therefore
tries Torch first (`OPENWARP_HAS_TORCH`, `TORCH_ROOT`), then ORT, then
mock. ORT paths stay compiled when present — nothing is removed.

Naming gotcha (verified 2026-10-25 with `curl -I`, redirect followed):
2.9.1 has NO `libtorch-cxx11-abi-shared-with-deps-…` build (that infix
404s — cxx11 ABI is now the only build, so the infix is gone). The exact
working URL pattern is:

```
https://download.pytorch.org/libtorch/<flavor>/libtorch-shared-with-deps-2.9.1%2B<flavor>.zip
```

Availability + size (HEAD, final 200 after the R2 redirect):

| flavor | URL suffix | size | use |
|--------|-----------|------|-----|
| `cu126` | `cu126/libtorch-shared-with-deps-2.9.1%2Bcu126.zip` | ~3.0 GiB | **want this** — matches the `requirements-cu126.txt` venv |
| `cu128` | `cu128/libtorch-shared-with-deps-2.9.1%2Bcu128.zip` | ~3.6 GiB | fallback if cu126 ever lacks a build (libtorch bundles its own CUDA, venv version need not match) |
| `cpu` | `cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip` | ~172 MiB | CPU-only fallback / version-match check |

TorchScript version match is strict: the `.pt` must be scripted by the
SAME torch (2.9.1) as the libtorch that loads it — export on the 3060
venv (§3c validation), never copy a `.pt` across torch versions.

3060 setup (do NOT download on dev/ARM boxes — multi-GiB, and there is
no official Linux-aarch64 libtorch zip anyway):

```
cd ~/opt
curl -LO https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.1%2Bcu126.zip
unzip -q libtorch-shared-with-deps-2.9.1%2Bcu126.zip   # yields ~/opt/libtorch/
ml/.venv/bin/python ml/export_script.py                # writes ml/weights/fill_blur_init.pt, reload-checks it
./build.sh --torch-root ~/opt/libtorch                 # or TORCH_ROOT=~/opt/libtorch
```

`build.sh` also auto-detects `~/opt/libtorch*` and `/opt/libtorch*`
(`share/cmake/Torch/TorchConfig.cmake` must exist) and prints
`Torch C++ : <dir>` in the summary. Configure must say `libtorch
found - NN fill uses TorchScript (preferred over ORT when both link)`.
No CUDA staging is done for torch (and none must be added): its bundled
libs already carry `$ORIGIN` runpaths — do NOT copy/symlink anything
into the libtorch dir.

3060 validation (after the build):

```
ml/.venv/bin/python ml/export_script.py   # SCRIPT EXPORT: PASS, max-abs-diff vs eager 0
./build/openwarp-vulkan-test              # still PASS (Torch absent here does not change its path)
OPENWARP_NN_LOG=1 ./build/openwarp --vulkan   # press T to NN fill; want backend=TORCH[cuda,cpu]
```

Tuning knobs (same env pattern as ORT): `OPENWARP_NN_THREADS=N` (torch
CPU thread pool, default = core count; logged at Load),
`OPENWARP_NN_CUDA_DEVICE=0`. fp32 for now — fp16 is a follow-up.

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
- Test files, `datasetgen/`, `data.py`, `mise.toml`, requirement files,
  and the committed `ml/weights/fill_blur_init.onnx` are untouched.
- Python delta is additive only: new `ml/export_script.py` (TorchScript
  twin export + reload check) and an optional `--script` flag on
  `ml/export_blur.py` that also saves the scripted twin (default
  behavior unchanged).
- Failure safety kept: any load/infer failure latches black
  (`DisableToBlack`), never garbage; the early-out fires only on
  exactly-zero holes (a `~0` threshold would leave black specks and
  order-dependent output).
