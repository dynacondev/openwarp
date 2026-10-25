# Tiny hole-fill baseline: 6ch in -> RGB out, blur-initialized, no training.
#
# Input channels (half-res, NCHW, float32 0..1):
#   0..2  warped RGB (holes = 0)
#   3      splat depth 0..1 (holes = 0)
#   4      hole mask (0 = valid, 1 = hole)
#   5      previous filled frame, luma only, reprojected by pose delta
# Output: predicted RGB for hole pixels. The caller composites with a hard
# mask (out = mask ? pred : warped), so valid pixels stay bit-exact.
#
# Part of the Openwarp project, NCSA licensed (see LICENSE).

import torch
import torch.nn as nn


class GatedConv2d(nn.Module):
    """Feature conv * sigmoid(gate conv). Gate bias init decides openness."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, gate_bias: float = 2.0):
        super().__init__()
        pad = kernel // 2
        self.feature = nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=True)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=True)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x) * torch.sigmoid(self.gate(x))


class TinyFillNet(nn.Module):
    """Single-scale 4-conv net, 12ch hidden, receptive field 9x9 at half-res.

    Kept fully convolutional with no stride so blur-weight init reproduces
    a box blur exactly (all activations stay non-negative on 0..1 inputs).
    The trained UNet (with downsampling + skips) lands later in this file;
    architecture choice is benchmarked via this net first.
    """

    def __init__(self, in_ch: int = 6, hidden: int = 12):
        super().__init__()
        self.in_ch = in_ch
        self.hidden = hidden
        self.gated = GatedConv2d(in_ch, hidden, kernel=3, gate_bias=2.0)
        self.body1 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=True)
        self.body2 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=True)
        self.head = nn.Conv2d(hidden, 3, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gated(x)
        x = self.relu(self.body1(x))
        x = self.relu(self.body2(x))
        return self.head(x)


def init_blur_weights(model: TinyFillNet) -> None:
    """Hand-set weights so the untrained net ~= 3x3 box blur of warped RGB.

    - Feature conv: output channel i averages input RGB channel (i % 3)
      over 3x3 (1/9 each); depth/mask/prev taps stay 0 (ignored for now).
    - Gate conv: all-zero weights, bias +2 -> sigmoid ~= 0.88 pass-through.
    - Body convs: Dirac (center tap 1 on same channel) -> identity on
      non-negative activations.
    - Head: output RGB j averages the hidden channels with (i % 3 == j).
    End-to-end on valid pixels ~= box-blurred warped RGB; holes (zeros in)
    get a local average once trained compositing uses the hard mask.
    """
    with torch.no_grad():
        f = model.gated.feature
        nn.init.zeros_(f.weight)
        nn.init.zeros_(f.bias)
        for o in range(model.hidden):
            f.weight[o, o % 3, :, :] = 1.0 / 9.0
        g = model.gated.gate
        nn.init.zeros_(g.weight)
        nn.init.constant_(g.bias, 2.0)
        for conv in (model.body1, model.body2):
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            for c in range(model.hidden):
                conv.weight[c, c, 1, 1] = 1.0
        h = model.head
        nn.init.zeros_(h.weight)
        nn.init.zeros_(h.bias)
        per_group = model.hidden // 3
        assert model.hidden % 3 == 0, "hidden must divide by 3 for blur init"
        for o in range(3):
            for k in range(per_group):
                h.weight[o, o + 3 * k, 0, 0] = 1.0 / per_group


def composite(warped_rgb: torch.Tensor, pred_rgb: torch.Tensor,
              hole_mask: torch.Tensor) -> torch.Tensor:
    """Hard-mask composite: valid pixels pass through bit-exact."""
    return torch.where(hole_mask > 0.5, pred_rgb, warped_rgb)
