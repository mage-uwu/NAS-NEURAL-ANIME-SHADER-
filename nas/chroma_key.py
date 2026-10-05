"""Remove the magenta chroma background from a style reference and write an RGBA PNG."""
import sys
import numpy as np
from PIL import Image

KEY = np.array([228.0, 3.0, 210.0])


def chroma_key(rgb):
    rgb = rgb.astype(np.float32)
    # "Magenta-ness": how far R and B both sit above G. ~210 on the backdrop, <= 0 on most of the character.
    m = np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1]
    alpha = np.clip(1.0 - (m - 40.0) / (170.0 - 40.0), 0.0, 1.0)

    # Unmix the backdrop from partially covered pixels: c = a*fg + (1-a)*key.
    a = np.maximum(alpha, 1e-3)[..., None]
    fg = (rgb - (1.0 - a) * KEY) / a
    # Despill: never let R and B both exceed G where magenta leaked in.
    spill = np.clip(np.minimum(fg[..., 0], fg[..., 2]) - fg[..., 1], 0, None)
    fg[..., 0] -= spill
    fg[..., 2] -= spill
    fg = np.clip(fg, 0, 255)
    return np.dstack([fg, alpha * 255]).astype(np.uint8)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out = chroma_key(np.asarray(Image.open(src).convert("RGB")))
    Image.fromarray(out, "RGBA").save(dst)
    a = out[..., 3]
    print(f"{dst}: {out.shape[1]}x{out.shape[0]}, opaque {np.mean(a > 250):.1%}, partial {np.mean((a > 5) & (a <= 250)):.1%}")
