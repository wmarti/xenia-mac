# Linux ARM64 Build Fixes

This document tracks all the changes made to support building Xenia on Linux ARM64.

## 1. Documentation Updates

### docs/building.md
- Added `sudo apt update` before package installation
- Removed `libunwind-dev` from dependencies (conflicts with `libunwind-18-dev`)
- Added `clang` to the package list
- Added `libxcb1-dev` and `libxcb-xkb-dev` for X11/XCB support (required for GTK file picker)

**Fixed command:**
```bash
sudo apt update
sudo apt-get install clang libgtk-3-dev libpthread-stubs0-dev liblz4-dev libx11-dev libx11-xcb-dev libvulkan-dev libsdl2-dev libiberty-dev libc++-dev libc++abi-dev
```

## 2. Build System Fixes

### tools/build/premake
- Fixed `has_bin()` function to actually test if binary can execute on current platform
- Now tries to run the binary with `--version` to detect architecture mismatches
- Automatically rebuilds premake5 for Linux when needed

### premake5.lua
- Fixed platform filter patterns for Linux (lines 218-222)
- Changed from `Linux-*-ARM64` to `Linux-ARM64` 
- Changed from `Linux-*-x86_64` to `Linux-x86_64`
- Added `filter({})` to reset filter after platform setup

### xb
- Updated build configuration to include architecture suffix (lines 914-929)
- Now detects system architecture and uses correct config format
- Example: `debug_linux-arm64` instead of just `debug_linux`

## 3. Code Fixes

### src/xenia/base/string_util.h
- Fixed `to_hex_string(uintptr_t)` redefinition error on ARM64 Linux
- Issue: On Linux ARM64, `uintptr_t` and `uint64_t` are the same type (both `unsigned long`)
- Solution: Use conditional compilation to only define the overload when types are distinct
- Added type traits include and proper platform detection

**Solution implemented:**
- Used SFINAE (Substitution Failure Is Not An Error) with `std::enable_if` 
- Template function that only instantiates when `uintptr_t` is distinct from both `uint32_t` and `uint64_t`
- This avoids preprocessor limitations and provides compile-time type checking

### src/xenia/cpu/backend/a64/a64_op.h
- Fixed deprecated enum-enum arithmetic warnings
- Issue: Arithmetic between different enumeration types is deprecated in C++20
- Solution: Cast enum values to their underlying type before arithmetic operations
- Lines affected: 39-45, 68, 71, 75, 79

## 4. FFmpeg Build Configuration

### third_party/FFmpeg/generate_premake.py
- Added Linux ARM64 configuration support (line 40)
- Fixed Linux x86_64 platform filter from `platforms:Linux` to `platforms:Linux-x86_64` (line 39)
- Added macOS ARM64 configuration support (line 41)
- Added: `Config('linux', 'aarch64', 'config_linux_aarch64.h', 'platforms:Linux-ARM64')`
- Added: `Config('macos', 'aarch64', 'config_macos_aarch64.h', 'platforms:Mac')`

### third_party/FFmpeg/config_linux_aarch64.h
- Created by copying config_macos_aarch64.h as a starting point
- Provides ARM64 Linux-specific FFmpeg configuration

### third_party/FFmpeg/libavutil/premake5.lua (regenerated)
- Now properly filters ARM64 and x86_64 architectures for Linux
- Line 198: ARM64 files included for `platforms:Linux-ARM64`
- Line 213: x86 files included only for `platforms:Linux-x86_64`

## 5. GTK Build Configuration

### src/xenia/ui/premake5.lua
- Added Linux-specific GTK configuration using pkg-config (lines 22-29)
- Uses `pkg-config --cflags gtk+-3.0` for include paths
- Uses `pkg-config --libs gtk+-3.0` for linking
- Fixes missing gdk/gdkx.h header issue

## 6. A64 Code Cache for Linux

### src/xenia/cpu/backend/a64/a64_code_cache_posix.cc
- Created POSIX-compatible version that works for both Linux and macOS
- Uses conditional compilation for platform-specific features:
  - macOS: `pthread_jit_write_protect_np()` and `sys_icache_invalidate()`
  - Linux: Direct memory copy and `__builtin___clear_cache()`
- Handles instruction cache flushing appropriately for each platform

## Known Issues to Fix

1. ~~Cannot use `std::is_same_v` in preprocessor directives~~ ✓ Fixed with SFINAE
2. ~~Deprecated enum arithmetic in a64_op.h~~ ✓ Fixed with static_cast
3. ~~A64CodeCache::Create() undefined reference~~ ✓ Fixed with a64_code_cache_posix.cc
4. ~~FFmpeg building x86 code on ARM64 Linux~~ ✓ Fixed with proper platform filters

## Platform Differences

### Type definitions on ARM64:
- **Linux ARM64**: `uint64_t` → `unsigned long`, `uintptr_t` → `unsigned long` (same type)
- **macOS ARM64**: `uint64_t` → `unsigned long long`, `uintptr_t` → `unsigned long` (distinct types)

This causes ODR violations on Linux but not on macOS when both overloads are present.