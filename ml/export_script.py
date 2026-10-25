# Export the blur-initialized TinyFillNet to TorchScript + verify vs eager.
#
#   ml/.venv/bin/python ml/export_script.py [--out ml/weights/fill_blur_init.pt] [--h 512 --w 512]
#
# Uses torch.jit.script (not trace): the net's control flow is static, but
# script is the robust choice — it captures the module (GatedConv2d,
# constants) structurally instead of freezing one traced path.
#
# Verifies: the .pt reloads, its output matches eager EXACTLY (bitwise, or
# <1e-6 max-abs-diff to absorb any lowering noise), and the same valid-pixel
# box-blur MAE check as export_blur.py. Prints the torch version into the
# log line so the 3060 run can confirm 2.9.1 == 2.9.1 with libtorch.
#
# Part of the Openwarp project, NCSA licensed (see LICENSE).

import argparse

import numpy as np
import torch
from fill_net import TinyFillNet, init_blur_weights


def reference_box_blur(rgb: np.ndarray) -> np.ndarray:
    """3x3 box blur of HxWx3 RGB, edge-replicated. Matches blur-init intent."""
    padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros_like(rgb)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + rgb.shape[0], dx : dx + rgb.shape[1]]
    return acc / 9.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml/weights/fill_blur_init.pt")
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--w", type=int, default=512)
    args = ap.parse_args()

    print(f"[export-script] torch {torch.__version__}")
    torch.manual_seed(0)
    model = TinyFillNet(in_ch=6, hidden=12)
    init_blur_weights(model)
    model.eval()

    H, W = args.h, args.w
    scripted = torch.jit.script(model)
    scripted.save(args.out)
    print(f"[export-script] wrote {args.out} (torch {torch.__version__})")

    # --- reload + verify with synthetic warp-like input ---
    rng = np.random.default_rng(0)
    rgb = rng.random((H, W, 3), dtype=np.float32)
    depth = np.full((H, W, 1), 0.8, dtype=np.float32)
    mask = np.zeros((H, W, 1), dtype=np.float32)
    mask[H // 2 - 32 : H // 2 + 32, W // 2 - 32 : W // 2 + 32] = 1.0
    rgb_in = rgb * (1.0 - mask)  # holes are zero, like the pull pass
    prev = np.full((H, W, 1), 0.5, dtype=np.float32)
    x = np.concatenate([rgb_in, depth, mask, prev], axis=-1)
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    with torch.no_grad():
        eager = model(torch.from_numpy(x)).numpy()
    reloaded = torch.jit.load(args.out)
    reloaded.eval()
    with torch.no_grad():
        got = reloaded(torch.from_numpy(x)).numpy()
    assert got.shape == (1, 3, H, W), f"unexpected shape {got.shape}"
    assert np.all(np.isfinite(got)), "non-finite outputs"
    maxdiff = float(np.abs(got - eager).max())
    print(
        f"[export-script] reload shape {got.shape}, range "
        f"[{got.min():.4f}, {got.max():.4f}], max-abs-diff vs eager: {maxdiff:.3e}"
    )
    assert maxdiff < 1e-6, f"scripted output diverged from eager (maxdiff={maxdiff})"

    # Same valid-pixel box-blur MAE check as export_blur.py.
    pred = np.transpose(got[0], (1, 2, 0))
    ref = reference_box_blur(rgb_in)
    valid = mask[..., 0] < 0.5
    scale = float(1.0 / (1.0 + np.exp(-2.0)))  # gate sigmoid(2) attenuation
    mae = np.abs(pred[valid] / scale - ref[valid]).mean()
    print(f"[export-script] valid-pixel MAE vs box blur (gate-compensated): {mae:.5f}")
    assert mae < 0.02, f"blur init diverged (mae={mae})"

    print("SCRIPT EXPORT: PASS")


if __name__ == "__main__":
    main()
