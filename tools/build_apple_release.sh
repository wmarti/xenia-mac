#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/build_apple_release.sh [options]

Builds and packages Apple release artifacts:
- macOS arm64 DMG
- macOS x86_64 DMG
- iOS arm64 unsigned IPA

Options:
  --out DIR            Output directory (default: artifacts/release)
  --config NAME        checked|debug|release|valgrind (default: release)
  --ios-min VERSION    iOS minimum version (default: 16.0)
  --macos-min VERSION  macOS minimum version (default: 15.0)
  --skip-ios
  --skip-macos-arm64
  --skip-macos-x86_64
  -h, --help

Notes:
- iOS packaging creates an *unsigned* .ipa (for re-signing).
- This script expects Xcode command line tools (xcodebuild, codesign, hdiutil).
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_bin() {
  command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"
}

cap_config() {
  # checked -> Checked, debug -> Debug, release -> Release, valgrind -> Valgrind
  local s
  s="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [ -z "$s" ]; then
    return 0
  fi
  local first rest
  first="$(printf '%s' "$s" | cut -c1 | tr '[:lower:]' '[:upper:]')"
  rest="$(printf '%s' "$s" | cut -c2-)"
  printf '%s%s' "$first" "$rest"
}

repo_root() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd
}

find_first_app() {
  local dir="$1"
  local app=""
  app="$(find "$dir" -maxdepth 1 -type d -name "*.app" | LC_ALL=C sort | head -n 1 || true)"
  [ -n "$app" ] || return 1
  printf '%s' "$app"
}

resolve_macdeployqt() {
  if [ -n "${MACDEPLOYQT:-}" ] && [ -x "${MACDEPLOYQT}" ]; then
    echo "${MACDEPLOYQT}"
    return 0
  fi
  if command -v macdeployqt >/dev/null 2>&1; then
    command -v macdeployqt
    return 0
  fi
  local candidates=(
    "/opt/homebrew/opt/qt/bin/macdeployqt"
    "/usr/local/opt/qt/bin/macdeployqt"
  )
  for c in "${candidates[@]}"; do
    if [ -x "$c" ]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

adhoc_sign_macos_app() {
  local app_bundle="$1"
  local entitlements="$2"

  local main_exe="$app_bundle/Contents/MacOS/xenia_edge"
  local frameworks_dir="$app_bundle/Contents/Frameworks"
  local plugins_dir="$app_bundle/Contents/PlugIns"

  if [ -f "$main_exe" ]; then
    codesign --remove-signature "$main_exe" >/dev/null 2>&1 || true
  fi
  codesign --remove-signature "$app_bundle" >/dev/null 2>&1 || true

  if [ -d "$frameworks_dir" ]; then
    find "$frameworks_dir" -maxdepth 2 -name "*.framework" -type d -print0 | \
      xargs -0 -I{} codesign --remove-signature "{}" >/dev/null 2>&1 || true
    find "$frameworks_dir" -name "*.dylib" -type f -print0 | \
      xargs -0 -I{} codesign --remove-signature "{}" >/dev/null 2>&1 || true

    find "$frameworks_dir" -maxdepth 2 -name "*.framework" -type d -print0 | \
      xargs -0 -I{} codesign --force --sign - --timestamp=none "{}"
    find "$frameworks_dir" -name "*.dylib" -type f -print0 | \
      xargs -0 -I{} codesign --force --sign - --timestamp=none "{}"
  fi

  if [ -d "$plugins_dir" ]; then
    find "$plugins_dir" -name "*.dylib" -type f -print0 | \
      xargs -0 -I{} codesign --remove-signature "{}" >/dev/null 2>&1 || true
    find "$plugins_dir" -name "*.dylib" -type f -print0 | \
      xargs -0 -I{} codesign --force --sign - --timestamp=none "{}"
  fi

  if [ -f "$main_exe" ]; then
    codesign --force --sign - --timestamp=none "$main_exe"
  fi

  codesign --force --deep --timestamp=none --sign - \
    --entitlements "$entitlements" "$app_bundle"
  xattr -cr "$app_bundle"
}

package_macos_dmg() {
  local app_bundle="$1"
  local dmg_out="$2"
  local license_file="$3"

  rm -f "$dmg_out"
  local dmg_contents
  dmg_contents="$(mktemp -d "${TMPDIR:-/tmp}/xenia_dmg.XXXXXX")"
  mkdir -p "$dmg_contents"
  cp -R "$app_bundle" "$dmg_contents/"
  cp -f "$license_file" "$dmg_contents/" || true
  ln -s /Applications "$dmg_contents/Applications"
  hdiutil create -volname "Xenia-Edge" -srcfolder "$dmg_contents" -ov -format UDZO "$dmg_out"
  rm -rf "$dmg_contents"
}

package_ios_ipa_unsigned() {
  local app_bundle="$1"
  local ipa_out="$2"

  # Make the output absolute because we `cd` into a temp directory.
  local ipa_dir ipa_base
  ipa_dir="$(cd "$(dirname "$ipa_out")" && pwd)"
  ipa_base="$(basename "$ipa_out")"
  ipa_out="$ipa_dir/$ipa_base"

  rm -f "$ipa_out"
  local tmp
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/xenia_ipa.XXXXXX")"
  mkdir -p "$tmp/Payload"
  # ditto preserves bundle metadata better than cp -R on macOS.
  ditto "$app_bundle" "$tmp/Payload/$(basename "$app_bundle")"
  (cd "$tmp" && ditto -c -k --sequesterRsrc --keepParent "Payload" "$ipa_out")
  rm -rf "$tmp"
}

build_ios=1
build_macos_arm64=1
build_macos_x86_64=1
out_dir="artifacts/release"
config="release"
ios_min="16.0"
macos_min="15.0"

while [ $# -gt 0 ]; do
  case "$1" in
    --out)
      out_dir="${2:-}"; shift 2;;
    --config)
      config="${2:-}"; shift 2;;
    --ios-min)
      ios_min="${2:-}"; shift 2;;
    --macos-min)
      macos_min="${2:-}"; shift 2;;
    --skip-ios)
      build_ios=0; shift;;
    --skip-macos-arm64)
      build_macos_arm64=0; shift;;
    --skip-macos-x86_64)
      build_macos_x86_64=0; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      die "unknown argument: $1";;
  esac
done

root="$(repo_root)"
cd "$root"

if [ "$(uname -s)" != "Darwin" ]; then
  die "this script must be run on macOS"
fi

need_bin xcodebuild
need_bin codesign
need_bin hdiutil
need_bin ditto
need_bin xattr

mkdir -p "$out_dir"

buildcfg="$(cap_config "$config")"

echo "Config: $buildcfg"
echo "Output: $out_dir"
echo "macOS min: $macos_min"
echo "iOS min: $ios_min"

if [ "$build_macos_arm64" -eq 1 ]; then
  echo ""
  echo "== macOS arm64 =="
  ./xb build --config="$buildcfg" --arch=arm64 --target=xenia-app -- \
    MACOSX_DEPLOYMENT_TARGET="$macos_min"

  mac_dir="build/bin/Mac-ARM64/$buildcfg"
  app_bundle="$(find_first_app "$mac_dir")" || die "macOS arm64 app not found in $mac_dir"

  mkdir -p "$app_bundle/Contents/Resources"
  if [ -f assets/icon/xenia.icns ]; then
    cp -f assets/icon/xenia.icns "$app_bundle/Contents/Resources/"
  fi

  if macdeployqt="$(resolve_macdeployqt)"; then
    "$macdeployqt" "$app_bundle" -always-overwrite -verbose=2 \
      -libpath=third_party/metal-shader-converter/lib \
      -libpath=third_party/DirectXShaderCompiler/build_dxilconv_macos/lib \
      -libpath=/opt/homebrew/opt/sdl2/lib \
      -libpath=/usr/local/opt/sdl2/lib || true
  else
    echo "warning: macdeployqt not found; app may be missing Qt frameworks"
  fi

  if [ -f third_party/metal-shader-converter/lib/libmetalirconverter.dylib ]; then
    mkdir -p "$app_bundle/Contents/Frameworks"
    cp -f third_party/metal-shader-converter/lib/libmetalirconverter.dylib \
      "$app_bundle/Contents/Frameworks/" || true
  fi
  if [ -f third_party/DirectXShaderCompiler/build_dxilconv_macos/lib/libdxilconv.dylib ]; then
    mkdir -p "$app_bundle/Contents/Frameworks"
    cp -f third_party/DirectXShaderCompiler/build_dxilconv_macos/lib/libdxilconv.dylib \
      "$app_bundle/Contents/Frameworks/" || true
  fi

  adhoc_sign_macos_app "$app_bundle" "xenia.entitlements"
  package_macos_dmg "$app_bundle" "$out_dir/xenia_edge_macos_arm64.dmg" "LICENSE"
fi

if [ "$build_macos_x86_64" -eq 1 ]; then
  echo ""
  echo "== macOS x86_64 =="
  ./xb build --config="$buildcfg" --arch=x86_64 --target=xenia-app -- \
    MACOSX_DEPLOYMENT_TARGET="$macos_min"

  mac_dir="build/bin/Mac-x86_64/$buildcfg"
  app_bundle="$(find_first_app "$mac_dir")" || die "macOS x86_64 app not found in $mac_dir"

  mkdir -p "$app_bundle/Contents/Resources"
  if [ -f assets/icon/xenia.icns ]; then
    cp -f assets/icon/xenia.icns "$app_bundle/Contents/Resources/"
  fi

  if macdeployqt="$(resolve_macdeployqt)"; then
    "$macdeployqt" "$app_bundle" -always-overwrite -verbose=2 \
      -libpath=third_party/metal-shader-converter/lib \
      -libpath=third_party/DirectXShaderCompiler/build_dxilconv_macos_x86_64/lib \
      -libpath=/opt/homebrew/opt/sdl2/lib \
      -libpath=/usr/local/opt/sdl2/lib || true
  else
    echo "warning: macdeployqt not found; app may be missing Qt frameworks"
  fi

  if [ -f third_party/metal-shader-converter/lib/libmetalirconverter.dylib ]; then
    mkdir -p "$app_bundle/Contents/Frameworks"
    cp -f third_party/metal-shader-converter/lib/libmetalirconverter.dylib \
      "$app_bundle/Contents/Frameworks/" || true
  fi
  if [ -f third_party/DirectXShaderCompiler/build_dxilconv_macos_x86_64/lib/libdxilconv.dylib ]; then
    mkdir -p "$app_bundle/Contents/Frameworks"
    cp -f third_party/DirectXShaderCompiler/build_dxilconv_macos_x86_64/lib/libdxilconv.dylib \
      "$app_bundle/Contents/Frameworks/" || true
  fi

  adhoc_sign_macos_app "$app_bundle" "xenia.entitlements"
  package_macos_dmg "$app_bundle" "$out_dir/xenia_edge_macos_x86_64.dmg" "LICENSE"
fi

if [ "$build_ios" -eq 1 ]; then
  echo ""
  echo "== iOS arm64 (unsigned ipa) =="
  # Extra signal for Lua scripts (also used by CI).
  touch .ios_target

  ./xb premake --target_os=ios
  ./xb build --config="$buildcfg" --target_os=ios --no_premake --target=xenia-app -- \
    CODE_SIGNING_ALLOWED=NO \
    IPHONEOS_DEPLOYMENT_TARGET="$ios_min"

  ios_dir="build/bin/iOS-ARM64/$buildcfg"
  app_bundle="$(find_first_app "$ios_dir")" || die "iOS app not found in $ios_dir"
  package_ios_ipa_unsigned "$app_bundle" "$out_dir/xenia_edge_ios_arm64_unsigned.ipa"
fi

echo ""
echo "Done. Artifacts:"
find "$out_dir" -maxdepth 1 -type f \( -name "*.dmg" -o -name "*.ipa" \) -print
