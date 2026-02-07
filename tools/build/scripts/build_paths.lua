-- Robust iOS target detection.  The XE_TARGET_IOS env-var is set by
-- xenia-build.py and is the most reliable signal since it bypasses
-- premake option handling entirely.
function is_ios_target()
  return os.getenv("XE_TARGET_IOS") == "1"
      or (os.target() == "ios")
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
