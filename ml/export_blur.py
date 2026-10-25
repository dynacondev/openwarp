# Export the blur-initialized TinyFillNet to ONNX + verify with onnxruntime.
#
#   python3 ml/export_blur.py [--out ml/weights/fill_blur_init.onnx] [--h 512 --w 512]
#
# Validates: ONNX loads in ORT, output shape/range sane, blur-init matches
# a reference 3x3 box blur on valid pixels, and a CPU timing smoke test.
# Real latency numbers come from the 3060 host (see ml/README.md) - CPU
# timings here only prove the graph runs.
#
# Part of the Openwarp project, NCSA licensed (see LICENSE).

import argparse
import time

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


def check_blur_init(
    out: np.ndarray, rgb_in: np.ndarray, mask: np.ndarray, tag: str
) -> None:
    """Model output on valid pixels ~= reference 3x3 box blur of input RGB."""
    pred = np.transpose(out[0], (1, 2, 0))
    ref = reference_box_blur(rgb_in)
    valid = mask[..., 0] < 0.5
    # Gate sigmoid(2)=0.881 attenuates; compare up to that scale factor.
    scale = float(1.0 / (1.0 + np.exp(-2.0)))
    mae = np.abs(pred[valid] / scale - ref[valid]).mean()
    print(f"[{tag}] valid-pixel MAE vs box blur (gate-compensated): {mae:.5f}")
    assert mae < 0.02, f"blur init diverged (mae={mae})"


def run_torch_cuda(model: TinyFillNet, x: np.ndarray, H: int, W: int) -> np.ndarray:
    """fp32 correctness + fp32/fp16 CUDA timing, zero ORT involvement."""
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda unavailable — check nvidia-smi on this host")
    model.cuda().eval()
    xt = torch.from_numpy(x).cuda()
    with torch.no_grad():
        out = model(xt).float().cpu().numpy()
    # fp32 first: model.half() is in-place, so halve only after fp32 is done.
    for name, use_half in (("fp32", False), ("fp16", True)):
        m = model.half() if use_half else model
        inp = xt.half() if use_half else xt
        with torch.no_grad():
            for _ in range(10):
                _ = m(inp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 50
        with torch.no_grad():
            for _ in range(n):
                _ = m(inp)
        torch.cuda.synchronize()
        print(
            f"[torch-cuda] {name}: {(time.perf_counter() - t0) / n * 1000:.2f} "
            f"ms/infer at {W}x{H}"
        )
    return out


def run_torch_inductor(model: TinyFillNet, x: np.ndarray, H: int, W: int) -> np.ndarray:
    """Fused-kernel number: torch.compile(max-autotune) fp16, no new deps."""
    model.cuda().half().eval()
    xt = torch.from_numpy(x).cuda().half()
    print("[inductor] compiling (one-time Triton autotune, may take minutes)...")
    opt = torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        for _ in range(3):
            _ = opt(xt)
        out = opt(xt).float().cpu().numpy()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 50
    with torch.no_grad():
        for _ in range(n):
            _ = opt(xt)
    torch.cuda.synchronize()
    print(
        f"[inductor] fp16-fused: {(time.perf_counter() - t0) / n * 1000:.2f} "
        f"ms/infer at {W}x{H}"
    )
    return out


def run_ort_iobind(onnx_path: str, x: np.ndarray, H: int, W: int) -> np.ndarray:
    """ORT CUDA EP with device-resident tensors: no per-run H2D/D2H copies."""
    import onnxruntime as ort

    ort.preload_dlls()
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in sess.get_providers():
        raise SystemExit("CUDAExecutionProvider unavailable — see --ort cuda notes.")
    io = sess.io_binding()
    io.bind_ortvalue_input("input", ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0))
    y = ort.OrtValue.ortvalue_from_numpy(np.empty((1, 3, H, W), np.float32), "cuda", 0)
    io.bind_ortvalue_output("output", y)
    for _ in range(5):
        sess.run_with_iobinding(io)
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        sess.run_with_iobinding(io)
    print(
        f"[iobind] ORT-CUDA device-resident: "
        f"{(time.perf_counter() - t0) / n * 1000:.2f} ms/infer at {W}x{H}"
    )
    return y.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml/weights/fill_blur_init.onnx")
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument(
        "--ort",
        default="cpu",
        choices=["cpu", "cuda", "torch", "inductor", "iobind"],
        help="'cpu': ORT CPU EP; 'cuda': ORT CUDA EP (sync copies); "
        "'torch': torch CUDA eager fp32/fp16; "
        "'inductor': torch.compile fused fp16 (the number that matters); "
        "'iobind': ORT CUDA EP, device-resident tensors (no per-run copies)",
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TinyFillNet(in_ch=6, hidden=12)
    init_blur_weights(model)
    model.eval()

    H, W = args.h, args.w
    dummy = torch.zeros(1, 6, H, W)

    import onnx

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "H", 3: "W"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(args.out)
    onnx.checker.check_model(onnx_model)
    # Dynamo exporter parks initializers in a .data sidecar; inline them so
    # the artifact is a single self-contained file for ORT.
    for init in onnx_model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            del init.external_data[:]
            init.data_location = onnx.TensorProto.DEFAULT
    onnx.save(onnx_model, args.out)
    # onnx>=1.19 save() still emits a (redundant) `<name>.onnx.data` sidecar
    # even with all tensors inline — the file is proven self-contained, drop it.
    import os

    redundant = args.out + ".data"
    if os.path.exists(redundant):
        os.remove(redundant)
    print(
        f"[export] wrote {args.out} "
        f"({len(onnx_model.SerializeToString()) / 1024:.1f} KiB), opset {args.opset}"
    )

    # --- ORT verification with synthetic warp-like input ---
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    rgb = rng.random((H, W, 3), dtype=np.float32)
    depth = np.full((H, W, 1), 0.8, dtype=np.float32)
    mask = np.zeros((H, W, 1), dtype=np.float32)
    mask[H // 2 - 32 : H // 2 + 32, W // 2 - 32 : W // 2 + 32] = 1.0  # square hole
    rgb_in = rgb * (1.0 - mask)  # holes are zero, like the pull pass
    prev = np.full((H, W, 1), 0.5, dtype=np.float32)
    x = np.concatenate([rgb_in, depth, mask, prev], axis=-1)
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.ort in ("cuda", "iobind")
        else ["CPUExecutionProvider"]
    )
    if args.ort == "torch":
        # Immediate GPU number with no new deps: torch cu126 bundles its own
        # CUDA 12.6 user-space libs, so this works wherever the driver does.
        out = run_torch_cuda(model, x, H, W)
        tag = "torch-cuda"
    elif args.ort == "inductor":
        # Fused-kernel number: Triton kernels via torch.compile, no new deps
        # (triton ships with the cu126 torch wheel on x86_64).
        out = run_torch_inductor(model, x, H, W)
        tag = "inductor"
    elif args.ort == "iobind":
        # ORT CUDA EP, device-resident tensors: upload once, no per-run copies.
        out = run_ort_iobind(args.out, x, H, W)
        tag = "iobind"
    else:
        import onnxruntime as ort

        if args.ort == "cuda":
            # Loads the pip-installed CUDA 13 / cuDNN 9 libs (nvidia-* wheels
            # from onnxruntime-gpu[cuda,cudnn]) so the CUDA EP starts with no
            # system toolkit. torch (imported above) preloads its own cu126
            # libs; the two CUDA majors coexist in separate wheel dirs.
            ort.preload_dlls()
        sess = ort.InferenceSession(args.out, providers=providers)
        if args.ort == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
            raise SystemExit(
                "CUDAExecutionProvider unavailable — check nvidia-smi (driver with "
                "CUDA 13 support) and that setup-cuda installed "
                "onnxruntime-gpu[cuda,cudnn], then re-run setup-cuda."
            )
        out = sess.run(None, {"input": x})[0]
        tag = "ort"
    assert out.shape == (1, 3, H, W), f"unexpected shape {out.shape}"
    assert np.all(np.isfinite(out)), "non-finite outputs"
    print(
        f"[{tag}] output shape {out.shape}, range "
        f"[{out.min():.4f}, {out.max():.4f}]"
        + (f", providers {sess.get_providers()}" if args.ort in ("cpu", "cuda") else "")
    )

    check_blur_init(out, rgb_in, mask, tag)

    if args.ort in ("torch", "inductor", "iobind"):
        print("BLUR EXPORT: PASS")
        return

    # Timing (CPU number here is smoke only; the GPU number that matters is
    # measured on the 3060 host).
    for _ in range(5):
        sess.run(None, {"input": x})
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        sess.run(None, {"input": x})
    ms = (time.perf_counter() - t0) / n * 1000
    print(
        f"[ort] {args.ort.upper()} timing: {ms:.1f} ms/infer at {W}x{H} "
        f"(see README for 3060 numbers)"
    )

    print("BLUR EXPORT: PASS")


if __name__ == "__main__":
    main()
