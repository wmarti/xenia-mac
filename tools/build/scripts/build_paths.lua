-- Robust iOS target detection: checks both os.istarget("ios") (which
-- requires the premake binary to recognise "ios" as a valid --os value)
-- and the raw _OPTIONS["os"] command-line value as a fallback.
function is_ios_target()
  return os.istarget("ios") or (_OPTIONS["os"] == "ios")
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
