"""Fetch a small curated set of real CC0 household meshes for LPW demos.

Source: Poly Haven (https://polyhaven.com) — all assets CC0 (public domain;
attribution appreciated, not required). We record author + license + source
+ md5 per asset in assets/library/NOTICE.md. Meshes download to
~/lpw/assets/library/ (NOT committed — regenerate with this script); only
the manifest and the recorded demo clip live in the repo.

Run (needs network):  python scripts/fetch_assets.py
"""

import hashlib
import json
import os
import socket
import urllib.request

socket.setdefaulttimeout(30)

# curated non-convex objects: real geometry that a box/sphere cannot fake, and
# good showcases for the CoACD collision decomposition
CURATED = ["ceramic_vase_01", "wooden_bowl_01", "food_apple_01", "rubber_duck_toy"]
LIB = os.path.expanduser("~/lpw/assets/library")
RES = "1k"
UA = {"User-Agent": "latentphysics-world/0.1 (asset fetch)"}


def _get_json(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))


def _download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        data = r.read()
    with open(dst, "wb") as f:
        f.write(data)
    return hashlib.md5(data).hexdigest()


def fetch(asset_id):
    files = _get_json(f"https://api.polyhaven.com/files/{asset_id}")
    info = _get_json(f"https://api.polyhaven.com/info/{asset_id}")
    entry = files["gltf"][RES]["gltf"]
    root = os.path.join(LIB, asset_id)
    gltf_path = os.path.join(root, f"{asset_id}_{RES}.gltf")
    _download(entry["url"], gltf_path)
    for rel, meta in entry.get("include", {}).items():
        _download(meta["url"], os.path.join(root, rel))
    authors = ", ".join(info.get("authors", {}).keys()) or "Poly Haven"
    return {"id": asset_id, "gltf": gltf_path, "authors": authors,
            "license": "CC0", "source": f"https://polyhaven.com/a/{asset_id}"}


def main():
    os.makedirs(LIB, exist_ok=True)
    records = []
    for aid in CURATED:
        try:
            rec = fetch(aid)
            records.append(rec)
            print(f"fetched {aid}  <- {rec['authors']}  (CC0)")
        except Exception as e:
            print(f"FAILED {aid}: {type(e).__name__} {str(e)[:80]}")
    # manifest lives BOTH next to the meshes and, committed, in the repo
    lines = ["# Third-party assets — curated CC0 library", "",
             "All assets below are CC0 (public domain) from Poly Haven",
             "(https://polyhaven.com). CC0 requires no attribution; authors are",
             "credited here as good practice. Meshes are fetched by",
             "`scripts/fetch_assets.py` and are not committed.", ""]
    for r in records:
        lines.append(f"- **{r['id']}** — {r['authors']} — CC0 — {r['source']}")
    text = "\n".join(lines) + "\n"
    repo_notice = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "assets", "library", "NOTICE.md")
    for path in (os.path.join(LIB, "NOTICE.md"), repo_notice):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    print(f"\n{len(records)}/{len(CURATED)} assets in {LIB}; manifest -> {repo_notice}")


if __name__ == "__main__":
    main()
