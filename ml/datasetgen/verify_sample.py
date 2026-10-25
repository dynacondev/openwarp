#!/usr/bin/env python3
# ml/datasetgen/verify_sample.py — blur-baseline check on one dataset sample.
#
# Loads a sequence through ml/data.py, runs the committed
# ml/weights/fill_blur_init.onnx with onnxruntime (CPU), applies the hard
# mask composite (out = mask ? pred : warped), and reports:
#   * valid-pixel MAE of the raw net output vs the reference 3x3 box blur
#     (gate-compensated, same bar as ml/export_blur.py: < 0.02)
#   * valid-pixel passthrough exactness after the hard composite (== 0)
#   * hole-pixel MAE after the hard composite (informational: how far a
#     pure blur fill is from ground truth on this sample)
#
# Usage (from openwarp/):
#   ml/.venv/bin/python ml/datasetgen/verify_sample.py --data data \\
#       --split train --index 0
#
# Part of the Openwarp project, NCSA licensed (see LICENSE).

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "ml"))

from data import HoleFillDataset


def reference_box_blur(rgb):
    padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros_like(rgb)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + rgb.shape[0], dx : dx + rgb.shape[1]]
    return acc / 9.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    ds = HoleFillDataset(root=args.data, split=args.split)
    x, tgt, mask = ds[args.index]
    info = ds.sample_info(args.index)
    print(
        f"sample {args.split}/{ds.names[args.index]} "
        f"hole_frac={info['hole_frac']:.4f} "
        f"(512px: {info['hole_frac_512']:.4f})"
    )
    print(f"input {tuple(x.shape)} target {tuple(tgt.shape)} mask {tuple(mask.shape)}")
    assert tuple(x.shape) == (6, 512, 512)
    assert tuple(tgt.shape) == (3, 512, 512)

    wpath = args.weights or os.path.join(REPO, "ml", "weights", "fill_blur_init.onnx")
    import onnxruntime as ort

    sess = ort.InferenceSession(wpath, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()[None].astype(np.float32)})[0]
    assert out.shape == (1, 3, 512, 512) and np.all(np.isfinite(out))

    pred = np.transpose(out[0], (1, 2, 0))
    warped = np.transpose(x.numpy()[:3], (1, 2, 0))
    m = mask.numpy()
    valid = m < 0.5
    scale = float(1.0 / (1.0 + np.exp(-2.0)))
    # Reference uses warped RGB as-is (holes are 0, exactly the net input).
    ref = reference_box_blur(warped)
    mae_blur = float(np.abs(pred[valid] / scale - ref[valid]).mean())
    print(f"valid-pixel MAE vs box blur (gate-compensated): {mae_blur:.5f}")

    comp = np.where(m[..., None] > 0.5, pred, warped)
    mae_pass = float(np.abs(comp[valid] - warped[valid]).mean())
    true_full = np.transpose(tgt.numpy(), (1, 2, 0)) + warped * (1.0 - m[..., None])
    hole = ~valid
    mae_hole = float(np.abs(comp[hole] - true_full[hole]).mean())
    print(f"post-composite valid-pixel MAE vs warped (expect 0): {mae_pass:.6f}")
    print(f"post-composite hole-pixel MAE vs true (info): {mae_hole:.5f}")

    ok = mae_blur < 0.02 and mae_pass == 0.0
    print("BLUR SAMPLE CHECK: " + ("PASS" if ok else "FAIL"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
