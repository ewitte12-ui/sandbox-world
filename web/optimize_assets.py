"""Shrink the textures embedded in .glb models, in place.

Roughly 94% of this project's model payload is uncompressed PNG textures at far
higher resolution than a voxel game ever samples — chicken.glb is 21.3MB of PNG
inside a 22.6MB file. Downscaling them is by far the biggest lever on web
download size, and is invisible at the distances these animals are viewed from.

Runs against a COPY of the assets (web/dist/assets), never the repo originals,
so the desktop build keeps full-resolution textures.

Uses macOS's built-in `sips` for the actual resampling so there is no Pillow or
node dependency to install.

GLB layout: a 12-byte header followed by chunks — JSON, then BIN. Images live in
the BIN chunk and are addressed via bufferViews. Rewriting them means repacking
BIN and fixing every bufferView offset; accessors address data relative to their
bufferView, so they need no changes.

Usage:  optimize_assets.py <assets-dir> [max-pixels]
"""

import json
import os
import struct
import subprocess
import sys
import tempfile

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def pad4(n):
    return (4 - n % 4) % 4


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB")
    chunks, offset = {}, 12
    while offset < len(data):
        clen, ctype = struct.unpack_from("<II", data, offset)
        chunks[ctype] = data[offset + 8 : offset + 8 + clen]
        offset += 8 + clen + pad4(clen)
    return version, chunks


def write_glb(path, version, gltf, binary):
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * pad4(len(js))
    binary += b"\x00" * pad4(len(binary))
    total = 12 + 8 + len(js) + (8 + len(binary) if binary else 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, version, total))
        f.write(struct.pack("<II", len(js), JSON_CHUNK))
        f.write(js)
        if binary:
            f.write(struct.pack("<II", len(binary), BIN_CHUNK))
            f.write(binary)


def resize(png_bytes, max_px):
    """Downscale via sips. Returns original bytes if anything goes wrong."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "t.png")
        with open(src, "wb") as f:
            f.write(png_bytes)
        r = subprocess.run(
            ["sips", "-Z", str(max_px), src],
            capture_output=True,
        )
        if r.returncode != 0:
            return png_bytes
        with open(src, "rb") as f:
            out = f.read()
    # Never let "optimisation" make a file bigger.
    return out if len(out) < len(png_bytes) else png_bytes


def optimize(path, max_px):
    version, chunks = read_glb(path)
    gltf = json.loads(chunks[JSON_CHUNK].decode("utf-8"))
    binary = chunks.get(BIN_CHUNK, b"")
    views = gltf.get("bufferViews", [])
    if not views or not binary:
        return None

    # Pull every bufferView out as its own blob so BIN can be repacked after
    # images change length.
    blobs = [
        bytearray(binary[v.get("byteOffset", 0) : v.get("byteOffset", 0) + v["byteLength"]])
        for v in views
    ]

    for img in gltf.get("images", []):
        bv = img.get("bufferView")
        if bv is None or img.get("mimeType") != "image/png":
            continue
        blobs[bv] = bytearray(resize(bytes(blobs[bv]), max_px))

    # Repack, honouring the 4-byte alignment glTF requires.
    out = bytearray()
    for i, v in enumerate(views):
        out += b"\x00" * pad4(len(out))
        v["byteOffset"] = len(out)
        v["byteLength"] = len(blobs[i])
        out += blobs[i]

    if gltf.get("buffers"):
        gltf["buffers"][0]["byteLength"] = len(out)

    write_glb(path, version, gltf, bytes(out))
    return os.path.getsize(path)


def main():
    root = sys.argv[1]
    max_px = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    if subprocess.run(["which", "sips"], capture_output=True).returncode != 0:
        print("  sips not available — skipping texture downscale", file=sys.stderr)
        return

    total_before = total_after = 0
    for dirpath, _, names in os.walk(root):
        for name in sorted(names):
            if not name.endswith(".glb"):
                continue
            path = os.path.join(dirpath, name)
            before = os.path.getsize(path)
            try:
                after = optimize(path, max_px)
            except Exception as exc:  # noqa: BLE001 - never fail a build over this
                print(f"  {name}: skipped ({exc})")
                continue
            if after is None:
                continue
            total_before += before
            total_after += after
            print(f"  {name}: {before/1048576:.1f}MB -> {after/1048576:.1f}MB")

    if total_before:
        print(
            f"  models: {total_before/1048576:.1f}MB -> {total_after/1048576:.1f}MB "
            f"({100 * (1 - total_after / total_before):.0f}% smaller)"
        )


if __name__ == "__main__":
    main()
