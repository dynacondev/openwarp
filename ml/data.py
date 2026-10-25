# ml/data.py — PyTorch reader for the seeded hole-fill dataset.
#
# Layout (written by ml/datasetgen/gen_dataset.py):
#   data/<split>/seqNNNNNN__src.npz       {rgb: uint8 (512,512,3)}
#   data/<split>/seqNNNNNN__true_tm1.npz  {rgb: uint8 (512,512,3)}
#   data/<split>/seqNNNNNN__true_t.npz    {rgb: uint8 (512,512,3)}
#   data/<split>/seqNNNNNN__input_t.npz   {warped_rgb: uint8 (512,512,3),
#                                          depth_u16: uint16 (512,512),
#                                          mask: uint8 (512,512) 0/1,
#                                          prev: uint8 (512,512) luma}
#   data/<split>/seqNNNNNN__meta.json     poses, deltas, hole stats
#
# __getitem__ returns (input6ch, target_hole_rgb, mask), all float32:
#   input6ch: (6,512,512) 0..1 — ch0-2 warped RGB (holes 0), ch3 splat depth
#     (holes 0), ch4 hole mask 0/1, ch5 prev filled luma reprojected.
#   target_hole_rgb: (3,512,512) 0..1 — true frame RGB with valid pixels
#     zeroed (supervision lives on holes; the mask selects them).
#   mask: (512,512) 0/1 float32.
#
# Sequence-aware prev handling: training is i.i.d. over sequences with
# teacher-forced prev (ch5 = true_{t-1} reprojected). prev_mode='teacher'
# uses the stored frame; prev_mode='zero' substitutes zeros (t=0
# conditioning / ablation: what the net sees when no history exists yet).
# At inference the caller is expected to overwrite ch5 with its own
# previously filled frame.
#
# Part of the Openwarp project, NCSA licensed (see LICENSE).

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

HALF = 512


class HoleFillDataset(Dataset):
    def __init__(self, root="data", split="train", prev_mode="teacher"):
        if prev_mode not in ("teacher", "zero"):
            raise ValueError("prev_mode must be 'teacher' or 'zero'")
        self.root = root
        self.split_dir = os.path.join(root, split)
        self.prev_mode = prev_mode
        names = sorted(
            f[: -len("__src.npz")]
            for f in os.listdir(self.split_dir)
            if f.endswith("__src.npz")
        )
        if not names:
            raise RuntimeError(f"no sequences in {self.split_dir}")
        self.names = names

    def __len__(self):
        return len(self.names)

    def set_prev_mode(self, mode):
        if mode not in ("teacher", "zero"):
            raise ValueError("prev_mode must be 'teacher' or 'zero'")
        self.prev_mode = mode

    def sample_info(self, i):
        with open(os.path.join(self.split_dir, self.names[i] + "__meta.json")) as f:
            return json.load(f)

    def __getitem__(self, index):
        d = self.split_dir
        n = self.names[index]
        inp = np.load(os.path.join(d, n + "__input_t.npz"))
        true_rgb = np.load(os.path.join(d, n + "__true_t.npz"))["rgb"]
        warped = inp["warped_rgb"].astype(np.float32) / 255.0
        depth = inp["depth_u16"].astype(np.float32) / 65535.0
        mask = inp["mask"].astype(np.float32)
        prev = inp["prev"].astype(np.float32) / 255.0
        if self.prev_mode == "zero":
            prev = np.zeros_like(prev)
        x = np.concatenate(
            [warped, depth[..., None], mask[..., None], prev[..., None]], axis=-1
        )
        x = torch.from_numpy(np.transpose(x, (2, 0, 1)).copy())
        tgt = torch.from_numpy(
            np.transpose(
                (true_rgb.astype(np.float32) / 255.0) * mask[..., None], (2, 0, 1)
            ).copy()
        )
        return x, tgt, torch.from_numpy(mask.copy())
