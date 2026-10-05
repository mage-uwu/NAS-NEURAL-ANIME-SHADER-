"""Build the self-contained viewer page: embeds char_run.glb (base64) and the learned shader params.

Usage: python nas/build_viewer.py [--params nas/out/params.json] [--out viewer.html]
"""
import argparse
import base64
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--params", default=root / "nas/out/params.json")
ap.add_argument("--out", default=root / "viewer.html")
args = ap.parse_args()

page = (root / "nas/viewer_template.html").read_text()
page = page.replace("{{GLB_B64}}", base64.b64encode((root / "char_run.glb").read_bytes()).decode())
page = page.replace("{{NAS_SHADER_JS}}", (root / "nas/nas_shader.js").read_text())
page = page.replace("{{GENSHIN_JS}}", (root / "nas/genshin_shader.js").read_text())
page = page.replace("{{PARAMS}}", json.dumps(json.loads(Path(args.params).read_text())))
Path(args.out).write_text(page)
print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
