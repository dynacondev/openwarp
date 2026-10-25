# ML hole-fill spike (feat/ml)

Tiny blur-initialized fill net + ONNX export. Proves the runtime path
before any training effort. Toolchain: **mise** (python 3.12 + uv) and
**uv** (venv + packages). No system python/pip setup needed.

Pinned versions: python 3.12 (mise) · torch 2.9.1 (+cpu / +cu126) ·
onnx 1.23.2 · onnxscript 0.7.2 · onnxruntime(-gpu) 1.31.0 · opset 17.

## Layout

- `fill_net.py` — `TinyFillNet`: 6ch in → RGB out, single-scale, 12ch hidden
  (~8k params). First layer is gated conv; rest plain conv + ReLU.
  `init_blur_weights()` hand-sets weights so the untrained net ≈ 3×3 box
  blur of warped RGB (depth/mask/prev taps zeroed for now).
- `export_blur.py` — builds the net, exports `weights/fill_blur_init.onnx`
  (dynamic H/W, opset 17), verifies with onnxruntime: shape, finite range,
  valid-pixel MAE vs reference box blur, timing. `--ort cuda` selects the
  CUDA provider for the GPU timing.
- `weights/fill_blur_init.onnx` — generated artifact, committed so the 3060
  host can benchmark without exporting.
- `requirements-cpu.txt` / `requirements-cu126.txt` — pinned stacks, one per
  host. torch comes from the PyTorch wheel index, the rest from PyPI.

## Channels (6, one temporal)

| ch | content | note |
|----|---------|------|
| 0–2 | warped RGB, holes = 0 | pull output |
| 3 | splat depth 0..1, holes = 0 | |
| 4 | hole mask 0/1 | from splat alpha |
| 5 | prev filled luma, reprojected | only temporal input for now |

Output RGB is composited with the hard mask (`out = mask ? pred : warped`)
so valid pixels stay bit-exact — enforced in-shader at integration time,
`composite()` here is the reference.

## Setup

```
cd openwarp
mise install        # python 3.12 + uv, from openwarp/mise.toml
mise run setup-cpu  # this host (torch +cpu)
mise run setup-cuda # 3060 host (torch +cu126, onnxruntime-gpu)
```

## Run

```
mise run export                                   # 1280x1024, CPU EP
ml/.venv/bin/python ml/export_blur.py --ort cuda  # 3060 host, CUDA EP timing
```

Expected: `BLUR EXPORT: PASS` + MAE < 0.02 vs box blur. CPU timings here are
smoke only (ARM, ~56 ms); the number that matters is CUDA EP / TensorRT on
the 3060.

## 3060 host (Ubuntu 26) from scratch

```
sudo ubuntu-drivers autoinstall && sudo reboot
nvidia-smi                                   # 3060 visible, CUDA 12.x
curl https://mise.run | sh                   # then restart shell / activate mise
git clone <repo> && cd xplane-reproject/openwarp
git checkout feat/ml
mise install && mise run setup-cuda
ml/.venv/bin/python ml/export_blur.py --ort cuda
```

Optional TensorRT number (ships with TensorRT 10):

```
trtexec --onnx=ml/weights/fill_blur_init.onnx --fp16 \
  --shapes=input:2x6x1024x1280 --avgRuns=200
```

Target: whole fill (preprocess + infer + composite) comfortably under ~2 ms
at half-res stereo. If fp16 lands <1 ms, there is headroom for the trained
UNet (downsample levels + skips) that lands next.

## Next steps (not in this commit)

1. Benchmark on 3060, record ms.
2. Dataset: dump `(warped, depth, mask, prev, true)` pairs from the
   openwarp renderer harness.
3. Train: same `TinyFillNet` first (overfits least on small data), then UNet
   variant with skip connections + temporal warping loss.
4. Runtime: bake trained weights — TensorRT EP for NVIDIA now, hand-written
   Vulkan compute (`fillMode 1`) for the portable ship path.
