"""Fit the ultralight cel shader to a style reference.

The reference is a different pose and has content the 3D model lacks (the ribbons), so nothing is
matched pixel-to-pixel and nothing forces every reference color to appear. Each stage gets its own signal:
  - palette: not trained. Each model color anchor (k-means of the albedo) is assigned the nearest of the
             reference's dominant colors (k-means of the reference). Unmatched reference colors (ribbons)
             are simply never picked. Gradient-based palette fitting on unpaired data collapsed regions
             toward the reference's most common colors, so this is closed-form and frozen.
  - lines:   match line density + darkness, measured with a morphological black-hat (thin dark structures)
  - patch:   5x5 RGB patches of the final output should look like *some* reference patch

Usage: python nas/train.py [--data nas/data] [--ref refs/style_ref.png] [--steps 400] [--out nas/out]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from shader import CelShader, PALETTE_K

BG = 0.5  # neutral backdrop both sides are composited onto, so silhouette lines are comparable
PATCH = 5


def load_png(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255).permute(2, 0, 1)


def load_views(data_dir, pad=6):
    """Load G-buffers, cropped to the character's bounding box (+pad) to cut work ~4x."""
    views = json.loads((data_dir / "views.json").read_text())
    out = []
    for v in views:
        a = load_png(data_dir / f"{v['name']}_albedo.png")
        n = load_png(data_dir / f"{v['name']}_normal.png")
        d = load_png(data_dir / f"{v['name']}_depth.png")
        rows = torch.nonzero(a[3].amax(1) > 0)[:, 0]
        cols = torch.nonzero(a[3].amax(0) > 0)[:, 0]
        y0, y1 = max(0, int(rows.min()) - pad), min(a.shape[1], int(rows.max()) + pad + 1)
        x0, x1 = max(0, int(cols.min()) - pad), min(a.shape[2], int(cols.max()) + pad + 1)
        crop = lambda t: t[:, y0:y1, x0:x1]
        out.append({"name": v["name"], "albedo": crop(a[:3]), "alpha": crop(a[3:4]),
                    "normal": crop(n[:3]), "depth": crop(d[0:1])})
    return out


def kmeans(x, k, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


def char_height(alpha):
    rows = torch.nonzero(alpha[0].amax(1) > 0.5)
    return float(rows.max() - rows.min() + 1)


def load_reference(path, target_height):
    ref = load_png(path)
    rows = torch.nonzero(ref[3].amax(1) > 0.5)
    cols = torch.nonzero(ref[3].amax(0) > 0.5)
    ref = ref[:, rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    s = target_height / ref.shape[1]
    ref = F.interpolate(ref[None], scale_factor=s, mode="area")[0]
    return ref  # RGBA, character scaled to the same pixel height as the renders


def composite(rgb, alpha):
    return rgb * alpha + BG * (1 - alpha)


def sample_pixels(rgb, mask, n):
    pix = rgb.permute(1, 2, 0)[mask[0] > 0.9]
    idx = torch.randint(0, pix.shape[0], (n,))
    return pix[idx]


def sample_patches(rgb, mask, n):
    patches = F.unfold(rgb[None], PATCH, padding=PATCH // 2)[0].t()  # (H*W, 3*P*P)
    centers = torch.nonzero(mask[0].flatten() > 0.5)[:, 0]
    idx = centers[torch.randint(0, centers.shape[0], (n,))]
    return patches[idx]


def swd(x, y, n_proj=64):
    """Sliced Wasserstein-2 between two equal-size point sets."""
    proj = F.normalize(torch.randn(x.shape[1], n_proj), dim=0)
    px, _ = (x @ proj).sort(0)
    py, _ = (y @ proj).sort(0)
    return ((px - py) ** 2).mean()


def chamfer(x, bank, tau=0.002):
    """Mean soft-min squared distance from each row of x to its nearest rows in bank."""
    d = torch.cdist(x, bank) ** 2 / x.shape[1]
    return (-tau * torch.logsumexp(-d / tau, dim=1)).mean()


def blackhat(rgb, k=5):
    """Morphological black-hat on luma: closing(x) - x. Large on thin dark strokes, ~0 on flat fills."""
    l = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    dil = F.max_pool2d(F.pad(l, (k // 2,) * 4, mode="replicate"), k, 1)
    close = -F.max_pool2d(F.pad(-dil, (k // 2,) * 4, mode="replicate"), k, 1)
    return (close - l).clamp(min=0)


def line_stats(rgb, mask):
    """(soft fraction of stroke pixels, mean stroke strength) inside the mask."""
    bh = blackhat(rgb[None])[0, 0]
    m = mask[0] > 0.9
    stroke = torch.sigmoid((bh - 0.12) / 0.03)
    frac = stroke[m].mean()
    strength = (bh * stroke)[m].sum() / (stroke[m].sum() + 1e-6)
    return torch.stack([frac, strength])


def shade(model, v):
    return model(v["albedo"][None], v["alpha"][None], v["normal"][None], v["depth"][None])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="nas/data")
    ap.add_argument("--ref", default="refs/style_ref.png")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="nas/out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    views = load_views(Path(args.data))
    held_out = [v for v in views if v["name"].startswith("f5_")]
    train = [v for v in views if not v["name"].startswith("f5_")]
    h = float(np.mean([char_height(v["alpha"]) for v in views]))
    ref = load_reference(args.ref, h)
    ref_rgb, ref_a = ref[:3], ref[3:4]
    ref_comp = composite(ref_rgb, ref_a)
    print(f"{len(train)} train / {len(held_out)} held-out views, character ~{h:.0f}px tall, ref {tuple(ref.shape[1:])}")

    albedo_pix = torch.cat([sample_pixels(v["albedo"], v["alpha"], 4000) for v in train])
    model = CelShader(anchors=kmeans(albedo_pix, PALETTE_K))
    ref_palette = kmeans(sample_pixels(ref_rgb, ref_a, 30000), 16)
    ref_lines = line_stats(composite(ref_rgb, ref_a), ref_a)
    print("reference palette:", (ref_palette * 255).round().int().tolist())
    print(f"reference lines: stroke fraction {ref_lines[0]:.3f}, strength {ref_lines[1]:.3f}")
    with torch.no_grad():
        nearest = torch.cdist(model.pal_src, ref_palette).argmin(1)
        model.pal_dst.copy_(ref_palette[nearest])
        model.pal_temp.fill_(-2.0)    # fairly crisp region assignment
        model.pal_detail.fill_(-1.0)  # keep ~27% of the texture's own variation
    for p_ in (model.pal_dst, model.pal_temp, model.pal_detail):
        p_.requires_grad_(False)
    for a_, b_ in zip(model.pal_src.tolist(), model.pal_dst.tolist()):
        print("palette", [round(x * 255) for x in a_], "->", [round(x * 255) for x in b_])
    opt = torch.optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad], lr=0.02)
    n_pix, n_patch = 2048, 1024

    for step in range(args.steps + 1):
        batch = random.sample(train, 4)
        loss_p = loss_l = 0.0
        patch_bank = sample_patches(ref_comp, ref_a, 2048)
        for v in batch:
            img, parts = model(v["albedo"][None], v["alpha"][None], v["normal"][None], v["depth"][None], return_parts=True)
            img, base = img[0], parts["base"][0]
            comp = composite(img, v["alpha"])
            loss_p = loss_p + chamfer(sample_patches(comp, v["alpha"], n_patch), patch_bank)
            loss_l = loss_l + ((line_stats(comp, v["alpha"]) - ref_lines) ** 2).sum()
        loss_p, loss_l = loss_p / len(batch), loss_l / len(batch)
        loss = loss_p + 2.0 * loss_l
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                ho = np.mean([float(((line_stats(composite(shade(model, v), v["alpha"]), v["alpha"]) - ref_lines) ** 2).sum())
                              for v in held_out])
            print(flush=True)
            print(f"step {step:4d}  patch {loss_p.item():.5f}  lines {float(loss_l):.5f}  held-out lines {ho:.5f}")

    params = model.export()
    (out / "params.json").write_text(json.dumps(params, indent=1))
    torch.save(model.state_dict(), out / "shader.pt")
    n_learned = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_palette = model.pal_src.numel() + model.pal_dst.numel() + 2
    print(f"saved {out/'params.json'} ({n_learned} trained + {n_palette} palette numbers)")

    # Contact sheet: albedo | NAS output for held-out views, plus the reference.
    with torch.no_grad():
        tiles = []
        for v in held_out:
            tiles.append(composite(v["albedo"], v["alpha"]))
            tiles.append(composite(shade(model, v), v["alpha"]))
        H = max(t.shape[1] for t in tiles)
        tiles = [F.pad(t, (4, 4, 0, H - t.shape[1]), value=BG) for t in tiles]
        r = F.interpolate(ref_comp[None], size=(H, int(ref_comp.shape[2] * H / ref_comp.shape[1])), mode="area")[0]
        sheet = torch.cat(tiles + [r], 2).clamp(0, 1)
        Image.fromarray((sheet.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(out / "held_out.png")
    print(f"saved {out/'held_out.png'}")


if __name__ == "__main__":
    main()
