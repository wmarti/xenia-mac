-- Robust iOS target detection.  Uses multiple fallback signals:
--   1. Sentinel file (.ios_target) created by CI before running premake
--   2. XE_TARGET_IOS env-var set by xenia-build.py
--   3. premake os.target() / os.istarget()
--   4. Raw _OPTIONS["os"] from command line
function is_ios_target()
  -- Note: premake changes its script directory while processing includes, so
  -- checking a relative ".ios_target" can be unreliable. Resolve relative to
  -- the main premake script location.
  local main_dir = _MAIN_SCRIPT_DIR
  local sentinel = (main_dir and path.join(main_dir, ".ios_target")) or ".ios_target"
  return os.isfile(sentinel) or os.isfile(".ios_target")
      or os.getenv("XE_TARGET_IOS") == "1"
      or (os.target and os.target() == "ios")
      or os.istarget("ios")
      or (_OPTIONS and _OPTIONS["os"] == "ios")
end

build_root = "build"
build_bin = build_root .. "/bin/%{cfg.platform}/%{cfg.buildcfg}"
build_gen = build_root .. "/gen/%{cfg.platform}/%{cfg.buildcfg}"
build_obj = build_root .. "/obj/%{cfg.platform}/%{cfg.buildcfg}"

build_tools = "tools/build"
build_scripts = build_tools .. "/scripts"
build_tools_src = build_tools .. "/src"

if os.istarget("android") then
  platform_suffix = "android"
elseif os.istarget("windows") then
  platform_suffix = "win"
elseif is_ios_target() then
  platform_suffix = "ios"
elseif os.istarget("macosx") then
  platform_suffix = "mac"
else
  platform_suffix = "posix"
end
