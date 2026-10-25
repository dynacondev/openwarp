#!/usr/bin/env python3
# ml/datasetgen/gen_dataset.py — seeded hole-fill training-data generator.
#
# Samples short sequences around the fixed demo-scene anchor, renders them
# with the REAL engine (ml/datasetgen/capture: ObjScene + demo shaders +
# VulkanWarp pull pass), gates on hole fraction, and packages browseable
# half-res (512x512) npz files plus per-seq + dataset manifests.
#
# Atomic sample = sequence: src frame at P0, true frames at P1_{t-1} and
# P1_t. input_t is the real pull-pass warp(P0 -> P1_t): warped RGB + splat
# depth + hole mask (from splat validity + pull alpha), plus prev_t =
# warp(true_{t-1} -> P1_t) luma (teacher forcing: the previous *true* frame
# reprojected by the small delta, standing in for the previous filled frame).
#
# Motion model: the warp-baseline delta (P0->P1_t, large) and the
# inter-frame delta (P1_{t-1}->P1_t, small 90Hz-like step) follow roughly the
# SAME translation/rotation direction with variance: the inter delta is
# sampled in its own per-axis ranges, sign-aligned per axis to the warp
# delta, then gated on cosine similarity (>= --min-cos) and on magnitude
# ordering (|inter| <= |warp|). Never opposing by construction.
#
# Determinism: one np.random.Generator per split, seeded from
# SeedSequence([seed, split_idx]); candidates are drawn, gated and captured
# in index order. Same (seed, flags) -> byte-identical manifests (JSON with
# sorted keys, no timestamps) and numerically identical npz. Render/warp
# pixels additionally depend on GPU/driver (documented in README); the
# RNG/pose portion is platform-independent.
#
# Usage (from openwarp/):
#   ml/.venv/bin/python ml/datasetgen/gen_dataset.py --out data \\
#       --seed 0 --train 1000 --val 200
# Smoke:
#   ml/.venv/bin/python ml/datasetgen/gen_dataset.py --out data \\
#       --seed 0 --train 4 --val 2
#
# Needs: ml/datasetgen/capture built (see README for the g++ line) and a
# running X server for the hidden GL window (e.g. Xvfb).
#
# Part of the Openwarp project, NCSA licensed (see LICENSE).

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # openwarp/
FULL = 1024          # render / warp resolution (OpenwarpApplication 1024x1024)
HALF = 512           # net resolution
DEPTH_SCALE = 16777215.0
SPLIT_EMPTY = 0xFFFFFFFF

# Fixed scene anchor = TestRun default start pose (testrun.cpp):
# startPos (-3, 1.5, 2), startOrientation AngleAxis(-pi/4, Y).
ANCHOR_POS = np.array([-3.0, 1.5, 2.0], dtype=np.float64)
_s = float(np.sin(-np.pi / 8))
_c = float(np.cos(np.pi / 8))
ANCHOR_QUAT = np.array([0.0, _s, 0.0, _c], dtype=np.float64)  # xyzw


# --- quaternion helpers (xyzw convention) -----------------------------------

def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qnorm(q):
    return q / np.linalg.norm(q)


def qrot(q, v):
    qv = np.array([v[0], v[1], v[2], 0.0])
    r = qmul(qmul(q, qv), qconj(q))
    return r[:3]


def qfrom_euler_deg(rx, ry, rz):
    """Delta quat dq = qx(rx) * qy(ry) * qz(rz), degrees. Right-multiplied
    onto the base orientation, i.e. the delta lives in the camera-local frame."""
    hx, hy, hz = np.deg2rad([rx, ry, rz]) / 2.0
    qx = np.array([np.sin(hx), 0.0, 0.0, np.cos(hx)])
    qy = np.array([0.0, np.sin(hy), 0.0, np.cos(hy)])
    qz = np.array([0.0, 0.0, np.sin(hz), np.cos(hz)])
    return qnorm(qmul(qmul(qx, qy), qz))


# --- candidate sampling ------------------------------------------------------

def sample_candidate(rng, a):
    """Draw one candidate (anchor pose + warp/inter deltas). Returns (rec, ok,
    reason): rec holds everything for shots.json + meta; ok=False means an
    analytic (pre-render) gate rejected it."""
    rec = {}
    rec["seq_seed"] = int(rng.integers(2 ** 63))
    anchor_pos = ANCHOR_POS + rng.uniform(-a.anchor_pos, a.anchor_pos)
    anchor_e = rng.uniform(-a.anchor_rot, a.anchor_rot)
    anchor_q = qnorm(qmul(ANCHOR_QUAT, qfrom_euler_deg(*anchor_e)))
    tw = rng.uniform(-a.warp_pos, a.warp_pos)
    rw = rng.uniform(-a.warp_rot, a.warp_rot)
    ti_raw = rng.uniform(-a.inter_pos, a.inter_pos)
    ri_raw = rng.uniform(-a.inter_rot, a.inter_rot)
    # Per-axis sign alignment: inter follows warp's direction with variance.
    # Deadband keeps the raw sign when the warp axis is ~zero.
    ti = np.where(np.abs(tw) < 1e-9, ti_raw,
                  np.copysign(np.abs(ti_raw), np.where(tw >= 0, 1.0, -1.0)))
    ri = np.where(np.abs(rw) < 1e-9, ri_raw,
                  np.copysign(np.abs(ri_raw), np.where(rw >= 0, 1.0, -1.0)))

    ntw, nti = float(np.linalg.norm(tw)), float(np.linalg.norm(ti))
    nrw, nri = float(np.linalg.norm(rw)), float(np.linalg.norm(ri))
    cos_t = float(np.dot(tw, ti) / (ntw * nti)) if ntw > 1e-6 and nti > 1e-9 else 1.0
    cos_r = float(np.dot(rw, ri) / (nrw * nri)) if nrw > 1e-6 and nri > 1e-9 else 1.0

    if nti > ntw:
        return rec, False, "mag-trans"
    if nri > nrw:
        return rec, False, "mag-rot"
    if cos_t < a.min_cos:
        return rec, False, "cos-trans"
    if cos_r < a.min_cos:
        return rec, False, "cos-rot"

    p1 = anchor_pos
    q1 = anchor_q
    dq_w = qfrom_euler_deg(*rw)
    dq_i = qfrom_euler_deg(*ri)
    q0 = qnorm(qmul(q1, qconj(dq_w)))
    p0 = p1 - qrot(q0, tw)
    qm = qnorm(qmul(q1, qconj(dq_i)))
    pm = p1 - qrot(qm, ti)

    rec.update({
        "anchor_pos": [float(v) for v in anchor_pos],
        "anchor_euler_deg": [float(v) for v in anchor_e],
        "warp_t": [float(v) for v in tw],
        "warp_r_deg": [float(v) for v in rw],
        "inter_t": [float(v) for v in ti],
        "inter_r_deg": [float(v) for v in ri],
        "cos_trans": cos_t,
        "cos_rot": cos_r,
        "warp_trans_norm": ntw,
        "inter_trans_norm": nti,
        "warp_rot_norm": nrw,
        "inter_rot_norm": nri,
        "p0": {"pos": [float(v) for v in p0],
               "quat": [float(v) for v in qnorm(q0)]},
        "ptm1": {"pos": [float(v) for v in pm],
                 "quat": [float(v) for v in qnorm(qm)]},
        "pt": {"pos": [float(v) for v in p1],
               "quat": [float(v) for v in qnorm(q1)]},
    })
    return rec, True, ""


# --- capture -----------------------------------------------------------------

CAND_RE = re.compile(
    r"CAND\s+(\d+)\s+hole=([0-9.eE+-]+)\s+splat_empty=([0-9.eE+-]+)\s+"
    r"alpha_extra=([0-9.eE+-]+)\s+accepted=([01])")


def run_capture(capture_bin, shots, rawdir, repodir, args):
    os.makedirs(rawdir, exist_ok=True)
    shots_path = os.path.join(rawdir, "shots.json")
    payload = [{"cand": s["cand"], "p0": s["rec"]["p0"],
                "ptm1": s["rec"]["ptm1"], "pt": s["rec"]["pt"]} for s in shots]
    with open(shots_path, "w") as f:
        json.dump(payload, f)
    cmd = [capture_bin, "--shots", shots_path, "--rawdir", rawdir,
           "--spvdir", os.path.join(repodir, "build", "openwarp-shaders"),
           "--resources", os.path.join(repodir, "resources"),
           "--hole-min", str(args.hole_min), "--hole-max", str(args.hole_max),
           "--width", str(FULL), "--height", str(FULL)]
    env = dict(os.environ)
    if args.display:
        env["DISPLAY"] = args.display
    if "DISPLAY" not in env:
        sys.exit("error: DISPLAY is not set (need Xvfb for the hidden GL window)")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(p.stdout[-4000:], file=sys.stderr)
        print(p.stderr[-4000:], file=sys.stderr)
        sys.exit(f"error: capture failed (exit {p.returncode})")
    gl_renderer, vk_device = "", ""
    results = {}
    for line in p.stdout.splitlines():
        if line.startswith("GL_RENDERER="):
            gl_renderer = line.split("=", 1)[1]
        elif line.startswith("VK_DEVICE="):
            vk_device = line.split("=", 1)[1]
            continue
        m = CAND_RE.search(line)
        if m:
            results[int(m.group(1))] = {
                "hole": float(m.group(2)),
                "splat_empty": float(m.group(3)),
                "alpha_extra": float(m.group(4)),
                "accepted": bool(int(m.group(5))),
            }
    if len(results) != len(shots):
        sys.exit(f"error: capture reported {len(results)}/{len(shots)} shots")
    return results, gl_renderer, vk_device


# --- packaging (1024 -> 512) -------------------------------------------------

def block_mean_u8(a):
    h, w, c = a.shape
    return np.round(a.reshape(h // 2, 2, w // 2, 2, c).mean(axis=(1, 3))).astype(np.uint8)


def package_seq(cand, rec, cap, rawdir, outdir, name):
    """Read raw 1024px bins, build 512px products, write 4 npz + return meta."""
    px = FULL * FULL
    pre = os.path.join(rawdir, f"c{cand:06d}")
    src = np.fromfile(pre + "_src_rgba.bin", np.uint8).reshape(FULL, FULL, 4)
    src_d = np.fromfile(pre + "_src_depth.bin", np.float32).reshape(FULL, FULL)
    warp = np.fromfile(pre + "_warp_rgba.bin", np.uint8).reshape(FULL, FULL, 4)
    splat = np.fromfile(pre + "_splat.bin", np.uint32).reshape(FULL, FULL)
    tm1 = np.fromfile(pre + "_tm1_rgba.bin", np.uint8).reshape(FULL, FULL, 4)
    t = np.fromfile(pre + "_t_rgba.bin", np.uint8).reshape(FULL, FULL, 4)
    prev = np.fromfile(pre + "_prev_rgba.bin", np.uint8).reshape(FULL, FULL, 4)

    # GL-native row 0 = frustum bottom -> flip once to top-row-first.
    src, tm1, t = np.flipud(src), np.flipud(tm1), np.flipud(t)
    warp, splat = np.flipud(warp), np.flipud(splat)
    prev, src_d = np.flipud(prev), np.flipud(src_d)

    hole1024 = (splat == SPLIT_EMPTY) | (warp[..., 3] < 128)
    valid = (~hole1024).astype(np.float64)[..., None]  # (H,W,1)

    def masked_block_mean(rgb_f, valid_f):
        num = (rgb_f * valid_f).reshape(HALF, 2, HALF, 2, -1).sum(axis=(1, 3))
        den = valid_f.reshape(HALF, 2, HALF, 2, -1).sum(axis=(1, 3))
        return num / np.maximum(den, 1.0)

    warped_rgb = masked_block_mean(warp[..., :3].astype(np.float64), valid)
    warped_rgb = np.round(np.clip(warped_rgb, 0, 255)).astype(np.uint8)

    d01 = np.where(splat == SPLIT_EMPTY, 0.0,
                   splat.astype(np.float64) / DEPTH_SCALE)
    depth512 = masked_block_mean(d01[..., None], valid)[..., 0]
    depth_u16 = np.round(np.clip(depth512, 0, 1) * 65535).astype(np.uint16)

    mask512 = (hole1024.reshape(HALF, 2, HALF, 2).sum(axis=(1, 3)) > 0).astype(np.uint8)

    luma = (0.299 * prev[..., 0].astype(np.float64) +
            0.587 * prev[..., 1].astype(np.float64) +
            0.114 * prev[..., 2].astype(np.float64))
    prev512 = np.round(luma.reshape(HALF, 2, HALF, 2).mean(axis=(1, 3))).astype(np.uint8)

    src512 = block_mean_u8(src[..., :3].copy())
    tm1512 = block_mean_u8(tm1[..., :3].copy())
    t512 = block_mean_u8(t[..., :3].copy())

    hole512 = float(mask512.mean())
    assert src512.shape == (HALF, HALF, 3)
    # Any-hole dilation can only grow the fraction, at most 4x (one hole
    # pixel per 2x2 block). Guards against flip/order bugs, not dilation.
    assert hole512 + 1e-9 >= cap["hole"], (hole512, cap["hole"])
    assert hole512 <= min(1.0, 4.0 * cap["hole"] + 1e-9), (hole512, cap["hole"])

    np.savez_compressed(os.path.join(outdir, f"{name}__src.npz"), rgb=src512)
    np.savez_compressed(os.path.join(outdir, f"{name}__true_tm1.npz"), rgb=tm1512)
    np.savez_compressed(os.path.join(outdir, f"{name}__true_t.npz"), rgb=t512)
    np.savez_compressed(os.path.join(outdir, f"{name}__input_t.npz"),
                        warped_rgb=warped_rgb, depth_u16=depth_u16,
                        mask=mask512, prev=prev512)

    meta = {
        "seq": name, "cand": cand, "seq_seed": rec["seq_seed"],
        "anchor_pos": rec["anchor_pos"],
        "anchor_euler_deg": rec["anchor_euler_deg"],
        "warp_t": rec["warp_t"], "warp_r_deg": rec["warp_r_deg"],
        "inter_t": rec["inter_t"], "inter_r_deg": rec["inter_r_deg"],
        "cos_trans": rec["cos_trans"], "cos_rot": rec["cos_rot"],
        "p0": rec["p0"], "ptm1": rec["ptm1"], "pt": rec["pt"],
        "hole_frac": cap["hole"], "splat_empty_frac": cap["splat_empty"],
        "alpha_extra_frac": cap["alpha_extra"], "hole_frac_512": hole512,
        "src_depth_min": float(src_d.min()), "src_depth_max": float(src_d.max()),
    }
    with open(os.path.join(outdir, f"{name}__meta.json"), "w") as f:
        json.dump(meta, f, sort_keys=True, indent=2)
        f.write("\n")
    return meta


# --- driver ------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="seeded hole-fill dataset generator")
    ap.add_argument("--out", default="data", help="output dir (default: data)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--hole-min", type=float, default=0.02)
    ap.add_argument("--hole-max", type=float, default=0.15)
    ap.add_argument("--min-cos", type=float, default=0.5,
                    help="min cosine similarity warp-vs-inter delta (trans+rot)")
    ap.add_argument("--anchor-pos", type=float, nargs=3, default=[0.4, 0.25, 0.4],
                    help="anchor translation half-extent xyz (m)")
    ap.add_argument("--anchor-rot", type=float, nargs=3, default=[4.0, 4.0, 4.0],
                    help="anchor euler half-extent xyz (deg)")
    ap.add_argument("--warp-pos", type=float, nargs=3, default=[0.10, 0.06, 0.10],
                    help="warp delta translation half-extent xyz (m)")
    ap.add_argument("--warp-rot", type=float, nargs=3, default=[1.5, 1.5, 1.5],
                    help="warp delta euler half-extent xyz (deg)")
    ap.add_argument("--inter-pos", type=float, nargs=3, default=[0.012, 0.008, 0.012],
                    help="inter delta translation half-extent xyz (m)")
    ap.add_argument("--inter-rot", type=float, nargs=3, default=[0.2, 0.2, 0.2],
                    help="inter delta euler half-extent xyz (deg)")
    ap.add_argument("--batch", type=int, default=32, help="shots per capture call")
    ap.add_argument("--max-factor", type=float, default=10.0,
                    help="max candidates = factor * needed, else abort")
    ap.add_argument("--keep-raw", action="store_true", help="keep 1024px raw bins")
    ap.add_argument("--display", default=None, help="X display (default: $DISPLAY)")
    ap.add_argument("--capture-bin", default=None)
    return ap.parse_args(argv)


def git_commit(repodir):
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repodir,
                           capture_output=True, text=True)
        return p.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None):
    a = parse_args(argv)
    a.anchor_pos = np.array(a.anchor_pos)
    a.anchor_rot = np.array(a.anchor_rot)
    a.warp_pos = np.array(a.warp_pos)
    a.warp_rot = np.array(a.warp_rot)
    a.inter_pos = np.array(a.inter_pos)
    a.inter_rot = np.array(a.inter_rot)

    outdir = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    rawdir = os.path.join(HERE, "raw")
    cap_bin = a.capture_bin or os.path.join(HERE, "capture")
    if not os.path.isfile(cap_bin):
        sys.exit(
            "error: capture binary missing. Build it with:\n"
            "  g++ -std=c++17 -O2 -DGLEW_NO_GLU -DGLEW_STATIC "
            "-Iinclude/eigen -Iinclude/glfw/include -Iinclude/glew-cmake/include "
            "-Isrc/openwarp ml/datasetgen/capture.cpp src/openwarp/util/obj.cpp "
            "build/libopenwarp_vulkan.a build/lib/libglewd.a "
            "build/include/glfw/src/libglfw3.a -lvulkan -lGL -lX11 -lXext "
            "-ldl -lm -lpthread -o ml/datasetgen/capture")
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(rawdir):
        shutil.rmtree(rawdir)
    os.makedirs(rawdir, exist_ok=True)

    manifest = {
        "tool": "gen_dataset.py",
        "seed": a.seed,
        "counts": {"train": a.train, "val": a.val},
        "flags": {
            "hole_min": a.hole_min, "hole_max": a.hole_max,
            "min_cos": a.min_cos,
            "anchor_pos": [float(v) for v in a.anchor_pos],
            "anchor_rot": [float(v) for v in a.anchor_rot],
            "warp_pos": [float(v) for v in a.warp_pos],
            "warp_rot": [float(v) for v in a.warp_rot],
            "inter_pos": [float(v) for v in a.inter_pos],
            "inter_rot": [float(v) for v in a.inter_rot],
        },
        "anchor": {"pos": [float(v) for v in ANCHOR_POS],
                   "quat_xyzw": [float(v) for v in ANCHOR_QUAT]},
        "frame": FULL, "net_half": HALF,
        "git_commit": git_commit(REPO),
        "gl_renderer": "", "vk_device": "",
        "splits": {"train": [], "val": []},
        "stats": {},
    }

    for split_idx, (split, need) in enumerate((("train", a.train), ("val", a.val))):
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, split_idx]))
        splitdir = os.path.join(outdir, split)
        os.makedirs(splitdir, exist_ok=True)
        accepted_metas = []
        rejected = {}
        candidates = 0
        max_candidates = int(max(need * a.max_factor, need + 8))
        seq_i = 0
        pend = []  # analytically-passed, not yet captured
        cand_id = 0
        while len(accepted_metas) < need:
            while len(pend) < min(a.batch, need - len(accepted_metas)):
                if candidates >= max_candidates:
                    sys.exit(
                        f"error: {split}: {candidates} candidates for "
                        f"{len(accepted_metas)}/{need} accepted "
                        f"(rejected {rejected}). Widen ranges/gate.")
                candidates += 1
                rec, ok, reason = sample_candidate(rng, a)
                if not ok:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                pend.append({"cand": cand_id, "rec": rec})
                cand_id += 1
            if not pend:
                continue
            results, glr, vkd = run_capture(cap_bin, pend, rawdir, REPO, a)
            manifest["gl_renderer"] = manifest["gl_renderer"] or glr
            manifest["vk_device"] = manifest["vk_device"] or vkd
            for s in pend:
                cap = results[s["cand"]]
                if not cap["accepted"]:
                    rejected["hole-gate"] = rejected.get("hole-gate", 0) + 1
                    continue
                name = f"seq{seq_i:06d}"
                meta = package_seq(s["cand"], s["rec"], cap, rawdir, splitdir, name)
                meta["split"] = split
                meta["seq_index"] = seq_i
                accepted_metas.append(meta)
                seq_i += 1
                # Raw 1024px bins are ~32 MiB/seq; drop them per batch so a
                # 1200-seq run never stages tens of GiB (unless --keep-raw).
                if not a.keep_raw:
                    pre = os.path.join(rawdir, f"c{s['cand']:06d}")
                    for suf in ("_src_rgba.bin", "_src_depth.bin",
                                "_warp_rgba.bin", "_splat.bin",
                                "_tm1_rgba.bin", "_tm1_depth.bin",
                                "_t_rgba.bin", "_prev_rgba.bin"):
                        try:
                            os.remove(pre + suf)
                        except OSError:
                            pass
                if len(accepted_metas) >= need:
                    break
            pend = []
        manifest["splits"][split] = accepted_metas
        holes = [m["hole_frac"] for m in accepted_metas]
        manifest["stats"][split] = {
            "candidates": candidates,
            "accepted": len(accepted_metas),
            "rejected": rejected,
            "hole_mean": float(np.mean(holes)) if holes else 0.0,
            "hole_min": float(np.min(holes)) if holes else 0.0,
            "hole_max": float(np.max(holes)) if holes else 0.0,
        }
        print(f"[{split}] accepted {len(accepted_metas)}/{need} "
              f"from {candidates} candidates, rejected={rejected}")
        if holes:
            print(f"[{split}] hole frac: min={min(holes):.4f} "
                  f"mean={sum(holes) / len(holes):.4f} max={max(holes):.4f}")

    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
        f.write("\n")
    if not a.keep_raw:
        shutil.rmtree(rawdir)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
