# ML hole-fill spike (feat/ml)

Tiny blur-initialized fill net + ONNX export. Proves the runtime path
before any training effort.

## Layout

- `fill_net.py` — `TinyFillNet`: 6ch in → RGB out, single-scale, 12ch hidden
  (~8k params). First layer is gated conv; rest plain conv + ReLU.
  `init_blur_weights()` hand-sets weights so the untrained net ≈ 3×3 box
  blur of warped RGB (depth/mask/prev taps zeroed for now).
- `export_blur.py` — builds the net, exports `weights/fill_blur_init.onnx`
  (dynamic H/W, opset 17), verifies with onnxruntime: shape, finite range,
  valid-pixel MAE vs reference box blur, CPU timing smoke.
- `weights/fill_blur_init.onnx` — generated artifact, committed so the 3060
  host can benchmark without exporting.

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

CPU-only host (this box):
```
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r ml/requirements.txt
```

3060 host (NVIDIA):
```
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install onnx onnxruntime-gpu numpy
```

## Run

```
cd openwarp
python3 ml/export_blur.py
```

Expected: `BLUR EXPORT: PASS`, plus a CPU timing line that is smoke only.

## Real latency (3060 host)

```
trtexec --onnx=ml/weights/fill_blur_init.onnx --fp16 \
  --shapes=input:2x6x1024x1280 --avgRuns=200
```

Target: whole fill (preprocess + infer + composite) comfortably under ~2 ms
at half-res stereo. If fp16 TRT is <1 ms, there is headroom for the trained
UNet (downsample levels + skips) that lands next.

TensorRT needs a static or profiled shape: use the half-res stereo shape
above (`2x6x1024x1280` for 2560×2048-ish eyes, adjust to exact eye size).

## Next steps (not in this commit)

1. Benchmark on 3060, record ms.
2. Dataset: dump `(warped, depth, mask, prev, true)` pairs from the
   openwarp renderer harness.
3. Train: same `TinyFillNet` first (overfits least on small data), then UNet
   variant with skip connections + temporal warping loss.
4. Runtime: bake trained weights — TensorRT EP for NVIDIA now, hand-written
   Vulkan compute (`fillMode 1`) for the portable ship path.
