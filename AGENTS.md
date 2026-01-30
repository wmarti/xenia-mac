# Progress

- Added `xenia.entitlements` and `assets/icon/xenia.icns` for macOS app bundle (pr-build-plumbing).
- Added DirectXShaderCompiler include path to Windows projects (xenia-app, xenia-gpu, xenia-gpu-d3d12, xenia-ui-d3d12) (pr-windows-arm64).
- Adjusted AMD64 MOVBE path to Windows-only to fix macOS x86_64 build (`src/xenia/apu/conversion.h`) (pr-macos-platform).
- Rebases completed for pr-build-plumbing, pr-windows-arm64, pr-linux-arm64, pr-thirdparty-updates, pr-macos-platform, pr-mac-qt-ui, pr-mac-gpu-deps against `origin/edge` with `./xb format` + `./xb lint --all` run per branch.
- Created `edge-integration-clean` from `origin/edge` and merged PR branches; resolved CI.yml conflicts to include Linux ARM64 + Windows ARM64 jobs and keep release artifacts for Linux x86_64/ARM64, macOS arm64/x86_64, Windows x86_64 (Windows ARM64 not in release).
- Fixed Asio include paths in `third_party/asio.lua` and `src/xenia/kernel/premake5.lua` (now using `third_party/asio/include`) to resolve macOS build failures; merged into `edge-integration-clean` and updated pr-thirdparty-updates.
- Switched Asio include to `sysincludedirs` in `src/xenia/kernel/premake5.lua` so Xcode searches it for `<asio.hpp>`; merged into `edge-integration-clean` and updated pr-thirdparty-updates.
- Added `pr-gdbstub-fix` to rename GDB stub signal enum values to avoid macro collisions on macOS; merged into `edge-integration-clean`.
- Added `pr-macos-link-fix` to avoid linking Vulkan libs on macOS (xenia-gpu-vulkan / xenia-ui-vulkan); merged into `edge-integration-clean`.
- Added macOS framework + SDL2 link settings in `pr-macos-link-fix` to resolve unresolved symbols from discord-rpc, Metal, and SDL on macOS; merged into `edge-integration-clean`.
- Synced `src/xenia/kernel/xam/xam_info.cc`, `src/xenia/emulator.{h,cc}`, and `src/xenia/kernel/xsocket.cc` back to `origin/edge` to restore in-process relaunch support and remove Windows compile errors.
- Adjusted `src/xenia/kernel/premake5.lua` Windows define filter to apply to Windows-ARM64 and Windows-x86_64.
- macOS build fixes: link `xenia-gpu-metal` / `xenia-ui-metal` in `src/xenia/app/premake5.lua`, and build `threading_posix.cc` instead of `threading_mac.cc` in `src/xenia/base/premake5.lua` to restore missing threading symbols.

# Pending

- GH Actions run 21530090307 (edge) currently building all platforms; monitor until completion and address any failures.
- After CI is green, keep PR branches squashed to single commits and confirm minimal merge conflicts vs `origin/edge`.
