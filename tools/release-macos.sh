#!/bin/bash
# Xenia-Canary macOS Release Packaging Script
# Creates a signed DMG for distribution
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
CONFIG="release"
ARCH="arm64"
IDENTITY="-"  # Ad-hoc signing
OUTPUT_DIR="$PROJECT_ROOT/dist"
SKIP_BUILD=false
TAG=""
TAG_UPDATE=false

# Enforce macOS 15.0+
export MACOSX_DEPLOYMENT_TARGET=15.0

APP_NAME="Xenia-Canary"
VOLUME_NAME="Xenia-Canary"

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Create a signed DMG for Xenia-Canary macOS distribution.

OPTIONS:
    --skip-build        Skip the build step, use existing app bundle
    --config <cfg>      Build config: debug or release (default: release)
    --arch <arch>       Architecture: arm64 or x86_64 (default: arm64)
    --identity <id>     Code signing identity (default: - for ad-hoc)
                        Use "Developer ID Application: ..." for notarization
    --tag <version>     Create a git tag with this version
    --tag-update        Force-update existing tag to HEAD
    --output <dir>      Output directory (default: dist/)
    -h, --help          Show this help

EXAMPLES:
    # Build and package with ad-hoc signing
    $(basename "$0")

    # Package existing build without rebuilding
    $(basename "$0") --skip-build

    # Create a tagged release
    $(basename "$0") --tag v1.0.0

    # Update existing tag and rebuild
    $(basename "$0") --tag v1.0.0 --tag-update
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --identity)
            IDENTITY="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --tag-update)
            TAG_UPDATE=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Determine build paths based on architecture
if [[ "$ARCH" == "arm64" ]]; then
    BUILD_DIR="$PROJECT_ROOT/build/bin/Mac-ARM64"
elif [[ "$ARCH" == "x86_64" ]]; then
    BUILD_DIR="$PROJECT_ROOT/build/bin/Mac-x86_64"
else
    echo "ERROR: Unknown architecture: $ARCH"
    exit 1
fi

CONFIG_CAPITALIZED="$(tr '[:lower:]' '[:upper:]' <<< ${CONFIG:0:1})${CONFIG:1}"
APP_PATH="$BUILD_DIR/$CONFIG_CAPITALIZED/$APP_NAME.app"
ICON_PATH="$PROJECT_ROOT/assets/icon/xenia.icns"

echo "=== Xenia-Canary macOS Release ==="
echo "Config:      $CONFIG"
echo "Arch:        $ARCH"
echo "Identity:    $IDENTITY"
echo "Output:      $OUTPUT_DIR"
echo ""

# Step 1: Build (unless skipped)
if [[ "$SKIP_BUILD" == "false" ]]; then
    echo "==> Building $APP_NAME ($CONFIG, $ARCH)..."
    cd "$PROJECT_ROOT"
    ./xb build --config="$CONFIG" --arch="$ARCH"
    echo ""
fi

# Check app exists
if [[ ! -d "$APP_PATH" ]]; then
    echo "ERROR: App bundle not found at $APP_PATH"
    echo "Run without --skip-build to build first."
    exit 1
fi

# Step 2: Code sign
echo "==> Code signing $APP_NAME.app..."
codesign --force --deep --sign "$IDENTITY" "$APP_PATH"
echo "Signed with identity: $IDENTITY"
echo ""

# Verify signing
echo "==> Verifying signature..."
codesign --verify --verbose "$APP_PATH"
echo ""

# Step 3: Create staging directory
echo "==> Creating DMG staging area..."
STAGING_DIR=$(mktemp -d)
trap "rm -rf '$STAGING_DIR'" EXIT

cp -R "$APP_PATH" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

# Copy volume icon
if [[ -f "$ICON_PATH" ]]; then
    cp "$ICON_PATH" "$STAGING_DIR/.VolumeIcon.icns"
fi

# Step 4: Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine DMG filename
if [[ -n "$TAG" ]]; then
    DMG_NAME="$APP_NAME-macOS-$ARCH-$TAG.dmg"
else
    DMG_NAME="$APP_NAME-macOS-$ARCH.dmg"
fi
DMG_PATH="$OUTPUT_DIR/$DMG_NAME"

# Remove existing DMG
rm -f "$DMG_PATH"

# Step 5: Create DMG
echo "==> Creating DMG: $DMG_NAME..."
hdiutil create \
    -volname "$VOLUME_NAME" \
    -srcfolder "$STAGING_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

echo ""

# Step 6: Git tagging (if requested)
if [[ -n "$TAG" ]]; then
    echo "==> Creating git tag: $TAG..."
    cd "$PROJECT_ROOT"
    
    if git rev-parse "$TAG" >/dev/null 2>&1; then
        if [[ "$TAG_UPDATE" == "true" ]]; then
            echo "Tag exists, updating to HEAD..."
            git tag -f "$TAG"
            echo "To push: git push origin $TAG --force"
        else
            echo "Tag $TAG already exists. Use --tag-update to move it."
        fi
    else
        git tag "$TAG"
        echo "Tag created. To push: git push origin $TAG"
    fi
    echo ""
fi

# Summary
echo "=== Done ==="
echo "DMG created: $DMG_PATH"
echo "Size: $(du -h "$DMG_PATH" | cut -f1)"
echo ""
echo "To upload to GitHub:"
echo "  1. Go to https://github.com/wmarti/xenia-mac/releases"
echo "  2. Create a new release with tag: ${TAG:-<version>}"
echo "  3. Upload: $DMG_PATH"
