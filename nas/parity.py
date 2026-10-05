"""Compare the GLSL compositor output (from nas/parity.mjs) with the PyTorch shader on the same G-buffers.

Usage: python nas/parity.py [view]   (run `node nas/parity.mjs <view>` first)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from shader import CelShader  # noqa: E402

root = Path(__file__).resolve().parent.parent
view = sys.argv[1] if len(sys.argv) > 1 else "f5_az35"
ld = lambda p: torch.from_numpy(np.asarray(Image.open(root / f"nas/data/{view}_{p}.png").convert("RGBA")).astype(np.float32) / 255).permute(2, 0, 1)
a, n, d = ld("albedo"), ld("normal"), ld("depth")

model = CelShader(anchors=torch.zeros(len(json.loads((root / "nas/out/params.json").read_text())["pal_src"]) // 3, 3))
model.load_state_dict(torch.load(root / "nas/out/shader.pt"))
with torch.no_grad():
    ref = model(a[None, :3], a[None, 3:], n[None, :3], d[None, 0:1])[0] * a[3:]  # premultiplied, like GLSL
ref = ref.permute(1, 2, 0).numpy()

H, W = ref.shape[:2]
gl = np.frombuffer((root / f"nas/out/parity_{view}.bin").read_bytes(), np.uint8).reshape(H, W, 4)[::-1].astype(np.float32) / 255
mask = a[3].numpy() > 0.5
diff = np.abs(gl[..., :3] - ref)[mask] * 255
print(f"{view}: {mask.sum()} character px | mean |diff| {diff.mean():.2f}/255 | p99 {np.percentile(diff, 99):.1f} | max {diff.max():.0f}")
Image.fromarray(np.concatenate([ref, gl[..., :3]], 1).clip(0, 1).__mul__(255).astype(np.uint8)).save(root / f"nas/out/parity_{view}.png")
