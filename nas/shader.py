"""Ultralight learned cel shader.

A fixed chain of graphics ops with a few dozen learnable parameters. Every op here has a
1:1 GLSL counterpart in nas/nas_shader.js (checked by nas/parity.py), so what trains is what ships.

Inputs (per pixel, all in [0, 1] as read from 8-bit G-buffers):
  albedo  RGB  sRGB base color
  alpha   1    coverage mask
  normal  RGB  view-space normal packed as n * 0.5 + 0.5
  depth   1    MeshDepthMaterial depth (tight near/far around the character)

Pipeline:
  1. palette   albedo -> soft-assign to K source anchors -> learned target colors (+ learned detail residual)
  2. cel       N.L -> hard-ish step (learned threshold/softness) -> mix(shadow tint, 1)
  3. rim       (1 - N.z) step on the lit side -> additive rim color
  4. lines     Sobel(depth, normal, albedo luma, alpha) -> weighted sum -> step -> line color
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PALETTE_K = 12


def sobel_mag(x):
    """Sobel gradient magnitude, summed over channels. x: (B, C, H, W)."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device)
    k = torch.stack([kx, kx.t()]).unsqueeze(1)  # (2, 1, 3, 3)
    c = x.shape[1]
    g = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k.repeat(c, 1, 1, 1), groups=c)
    g = g.view(x.shape[0], c, 2, *x.shape[2:])
    return torch.sqrt((g ** 2).sum(2) + 1e-8).sum(1, keepdim=True)


def luma(rgb):
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


class CelShader(nn.Module):
    def __init__(self, anchors=None):
        super().__init__()
        # 1. palette: source anchors live in the model's albedo space, targets in the style's palette
        if anchors is None:
            anchors = torch.rand(PALETTE_K, 3)
        self.register_buffer("pal_src", torch.as_tensor(anchors, dtype=torch.float32))
        self.pal_dst = nn.Parameter(self.pal_src.clone())
        self.pal_temp = nn.Parameter(torch.tensor(0.0))  # temp = 0.002 + 0.02 * sigmoid
        self.pal_detail = nn.Parameter(torch.tensor(0.0))  # sigmoid: 0 = flat fill, 1 = keep texture detail
        # 2. cel
        # Key light is fixed (art direction, upper-left-front in view space); the response to it is learned.
        self.register_buffer("light", F.normalize(torch.tensor([-0.5, 0.6, 0.6]), dim=0))
        self.cel_thr = nn.Parameter(torch.tensor(0.0))  # -0.25 + 0.3 * sigmoid: shadow on surfaces facing away
        self.cel_soft = nn.Parameter(torch.tensor(-2.0))  # softness = 0.02 + 0.2 * sigmoid(raw)
        self.shadow_tint = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))  # 0.75 + 0.17 * sigmoid -> light cel band
        # 3. rim
        self.rim_thr = nn.Parameter(torch.tensor(0.6))
        self.rim_color = nn.Parameter(torch.tensor([-3.0, -3.0, -3.0]))  # softplus -> >= 0
        # 4. lines
        # depth, normal, albedo luma, alpha. Luma is masked off: on unpaired data it learns to hatch texture noise.
        self.line_w = nn.Parameter(torch.tensor([1.0, 0.0, -20.0, 0.0]))  # softplus
        self.register_buffer("line_w_mask", torch.tensor([1.0, 1.0, 0.0, 1.0]))
        self.line_thr = nn.Parameter(torch.tensor(0.0))  # 0.25 + softplus: lines only on real edges
        self.line_soft = nn.Parameter(torch.tensor(0.0))  # softness = 0.05 + 0.3 * sigmoid: crisp lines
        self.line_color = nn.Parameter(torch.tensor([-1.0, -0.8, -1.0]))  # 0.5 * sigmoid -> dark
        self.line_opacity = nn.Parameter(torch.tensor(1.5))  # 0.6 + 0.4 * sigmoid

    def light_dir(self):
        return self.light

    # Constrained views of the raw parameters (what the shader actually uses).
    def c_pal_temp(self):
        return 0.002 + 0.02 * torch.sigmoid(self.pal_temp)

    def c_cel_thr(self):
        return -0.25 + 0.3 * torch.sigmoid(self.cel_thr)

    def c_line_thr(self):
        return 0.25 + F.softplus(self.line_thr)

    def c_line_soft(self):
        return 0.05 + 0.3 * torch.sigmoid(self.line_soft)

    def c_line_opacity(self):
        return 0.6 + 0.4 * torch.sigmoid(self.line_opacity)

    def palette(self, albedo):
        d = ((albedo.unsqueeze(1) - self.pal_src.view(1, -1, 3, 1, 1)) ** 2).sum(2)  # (B, K, H, W)
        w = F.softmax(-d / self.c_pal_temp(), dim=1)
        src = torch.einsum("bkhw,kc->bchw", w, self.pal_src)
        dst = torch.einsum("bkhw,kc->bchw", w, self.pal_dst)
        return dst + torch.sigmoid(self.pal_detail) * (albedo - src)

    def c_shadow_tint(self):
        return 0.75 + 0.17 * torch.sigmoid(self.shadow_tint)

    def c_rim_color(self):
        return F.softplus(self.rim_color)

    def c_line_w(self):
        return F.softplus(self.line_w) * self.line_w_mask

    def c_line_color(self):
        return 0.5 * torch.sigmoid(self.line_color)

    def forward(self, albedo, alpha, normal, depth, return_parts=False):
        n = F.normalize(normal * 2 - 1, dim=1)

        # 1. palette
        base = self.palette(albedo).clamp(0, 1)

        # 2. cel
        ndl = (n * self.light_dir().view(1, 3, 1, 1)).sum(1, keepdim=True)
        soft = 0.02 + 0.2 * torch.sigmoid(self.cel_soft)
        lit = torch.sigmoid((ndl - self.c_cel_thr()) / soft)
        tint = self.c_shadow_tint().view(1, 3, 1, 1)
        color = base * (tint + (1 - tint) * lit)

        # 3. rim
        rim = torch.sigmoid((1 - n[:, 2:3] - self.rim_thr) / 0.05) * lit
        color = color + rim * self.c_rim_color().view(1, 3, 1, 1)

        # 4. lines
        edges = torch.cat([sobel_mag(depth), sobel_mag(normal), sobel_mag(luma(albedo)), sobel_mag(alpha)], 1)
        e = (edges * self.c_line_w().view(1, 4, 1, 1)).sum(1, keepdim=True)
        line = torch.sigmoid((e - self.c_line_thr()) / self.c_line_soft()) * self.c_line_opacity()
        color = color * (1 - line) + self.c_line_color().view(1, 3, 1, 1) * line

        color = color.clamp(0, 1)
        if return_parts:
            return color, {"lit": lit, "line": line, "base": base}
        return color

    def export(self):
        """Plain-number parameters for the GLSL generator."""
        soft = 0.02 + 0.2 * torch.sigmoid(self.cel_soft)
        r = lambda t: [round(float(v), 5) for v in t.flatten()]
        return {
            "pal_src": r(self.pal_src),
            "pal_dst": r(self.pal_dst.detach().clamp(0, 1)),
            "pal_temp": float(self.c_pal_temp()),
            "pal_detail": float(torch.sigmoid(self.pal_detail)),
            "light": r(self.light_dir().detach()),
            "cel_thr": float(self.c_cel_thr()),
            "cel_soft": float(soft),
            "shadow_tint": r(self.c_shadow_tint().detach()),
            "rim_thr": float(self.rim_thr),
            "rim_color": r(self.c_rim_color().detach()),
            "line_w": r(self.c_line_w().detach()),
            "line_thr": float(self.c_line_thr()),
            "line_soft": float(self.c_line_soft()),
            "line_color": r(self.c_line_color().detach()),
            "line_opacity": float(self.c_line_opacity()),
        }
