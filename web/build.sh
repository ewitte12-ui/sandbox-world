#!/usr/bin/env bash
# Builds the WebGL2 (wasm) version into web/dist/.
#
# Requires rustup (Homebrew's rust ships only the host target and cannot add
# wasm32), plus a wasm-bindgen CLI whose version EXACTLY matches the
# wasm-bindgen crate in Cargo.lock — a mismatch fails at bindgen time with a
# schema-version error.
#
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version <version from Cargo.lock>
#
# Usage:  web/build.sh [--debug]
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="release"
PROFILE_FLAG="--release"
if [[ "${1:-}" == "--debug" ]]; then
  PROFILE="debug"
  PROFILE_FLAG=""
fi

TARGET_DIR="target/wasm32-unknown-unknown/${PROFILE}"
OUT="web/dist"

# Resolve the toolchain explicitly rather than trusting PATH. Homebrew's rust
# also provides a `cargo`, but it ships only the host target and cannot build
# wasm — and it wins on PATH in any shell started before rustup was installed.
CARGO="${CARGO:-}"
if [[ -z "$CARGO" ]]; then
  if [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    CARGO="$HOME/.cargo/bin/cargo"
  else
    CARGO="cargo"
  fi
fi

if ! "$CARGO" --version >/dev/null 2>&1; then
  echo "error: no usable cargo (tried '$CARGO')." >&2
  exit 1
fi

# Fail early and legibly if this cargo has no wasm std, rather than letting
# rustc emit a wall of "can't find crate for `core`" errors.
SYSROOT="$("$CARGO" rustc -- --print sysroot 2>/dev/null || true)"
if [[ -z "$SYSROOT" ]]; then
  SYSROOT="$(rustc --print sysroot 2>/dev/null || true)"
fi
if [[ -n "$SYSROOT" && ! -d "$SYSROOT/lib/rustlib/wasm32-unknown-unknown" ]]; then
  echo "error: '$CARGO' has no wasm32-unknown-unknown std (sysroot: $SYSROOT)." >&2
  echo "       Homebrew's rust cannot add targets. With rustup installed, run:" >&2
  echo "         rustup target add wasm32-unknown-unknown" >&2
  exit 1
fi

EXPECTED=$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/[",]/,""); print $3; exit}' Cargo.lock)

# The CLI must match the wasm-bindgen crate EXACTLY, and that version is a
# property of the project, not the machine — the Bevy 0.18 sibling project
# (~/Documents/claude/metalworld-bevy) pins 0.2.114 while this one pins 0.2.127.
# A single `cargo install`ed wasm-bindgen in ~/.cargo/bin can only satisfy one of
# them, so prefer a per-version install and leave ~/.cargo/bin alone:
#
#   cargo install wasm-bindgen-cli --version <ver> --root ~/.local/wasm-bindgen/<ver>
#
# Resolution order: $WASM_BINDGEN, then the versioned dir, then ~/.cargo/bin, then PATH.
VERSIONED="$HOME/.local/wasm-bindgen/$EXPECTED/bin/wasm-bindgen"
WB="${WASM_BINDGEN:-}"
if [[ -z "$WB" ]]; then
  if [[ -x "$VERSIONED" ]]; then
    WB="$VERSIONED"
  elif [[ -x "$HOME/.cargo/bin/wasm-bindgen" ]]; then
    WB="$HOME/.cargo/bin/wasm-bindgen"
  else
    WB="wasm-bindgen"
  fi
fi
if ! command -v "$WB" >/dev/null 2>&1 && [[ ! -x "$WB" ]]; then
  echo "error: wasm-bindgen not found (tried '$WB')." >&2
  echo "       Install with:" >&2
  echo "         cargo install wasm-bindgen-cli --version $EXPECTED --root \$HOME/.local/wasm-bindgen/$EXPECTED" >&2
  echo "       Or set WASM_BINDGEN=/path/to/wasm-bindgen" >&2
  exit 1
fi

ACTUAL=$("$WB" --version | awk '{print $2}')
if [[ "$EXPECTED" != "$ACTUAL" ]]; then
  echo "error: wasm-bindgen CLI is $ACTUAL but Cargo.lock pins $EXPECTED." >&2
  echo "       These must match exactly or the generated glue will not load." >&2
  echo "       Install the matching version WITHOUT clobbering the one in ~/.cargo/bin:" >&2
  echo "         cargo install wasm-bindgen-cli --version $EXPECTED --root \$HOME/.local/wasm-bindgen/$EXPECTED" >&2
  echo "       (this script picks that up automatically)" >&2
  exit 1
fi

echo "==> cargo build (${PROFILE}, wasm32-unknown-unknown)  [$CARGO]"
"$CARGO" build ${PROFILE_FLAG} --target wasm32-unknown-unknown

echo "==> wasm-bindgen $ACTUAL  [$WB]"
rm -rf "$OUT"
mkdir -p "$OUT"
"$WB" --no-typescript --target web \
  --out-dir "$OUT" \
  --out-name sandbox_world \
  "${TARGET_DIR}/sandbox_world.wasm"

echo "==> staging page + assets"
cp web/index.html "$OUT/"
# Bevy's web asset reader fetches over HTTP relative to the page, so assets
# must sit beside index.html exactly as they do beside the native binary.
cp -R assets "$OUT/"
find "$OUT" -name '.DS_Store' -delete

# Every model is loaded as .glb (see animals.rs); the .gltf twin is a leftover
# export that nothing references. Harmless on desktop, ~6MB of dead download on
# the web, so drop it from the bundle rather than from the repo.
find "$OUT/assets" -name '*.gltf' -delete

# Downscale embedded model textures. Operates on the staged copy only — the
# repo keeps full-resolution assets for the desktop builds. Set WEB_TEXTURE_MAX
# to change the cap, or 0 to skip.
TEX_MAX="${WEB_TEXTURE_MAX:-512}"
if [[ "$TEX_MAX" != "0" ]]; then
  echo "==> downscaling model textures (max ${TEX_MAX}px)"
  python3 web/optimize_assets.py "$OUT/assets" "$TEX_MAX"
fi

# wasm-opt (binaryen) roughly halves the payload. Optional: skipped with a
# warning rather than failing, so a fresh clone can still produce a build.
if command -v wasm-opt >/dev/null 2>&1; then
  echo "==> wasm-opt -Oz"
  BEFORE=$(du -h "$OUT/sandbox_world_bg.wasm" | awk '{print $1}')
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
    -o "$OUT/sandbox_world_bg.wasm.opt" "$OUT/sandbox_world_bg.wasm"
  mv "$OUT/sandbox_world_bg.wasm.opt" "$OUT/sandbox_world_bg.wasm"
  echo "    ${BEFORE} -> $(du -h "$OUT/sandbox_world_bg.wasm" | awk '{print $1}')"
else
  echo "==> wasm-opt not found — skipping (install binaryen to shrink the wasm)"
fi

echo
echo "Built ${OUT}:"
du -sh "$OUT"
du -h "$OUT/sandbox_world_bg.wasm" | awk '{print "  wasm: " $1}'
echo
echo "Serve it (module scripts and wasm need real HTTP, not file://):"
echo "  python3 -m http.server -d $OUT 8080"
