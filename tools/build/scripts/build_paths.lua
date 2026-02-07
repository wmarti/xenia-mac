-- Robust iOS target detection. Three checks for maximum compatibility:
--   1. os.target() == "ios" : raw target OS string set by --os=ios
--   2. os.istarget("ios")   : system-tag check (needs binary to know iOS)
--   3. _OPTIONS["os"]       : raw command-line option fallback
function is_ios_target()
  return (os.target() == "ios")
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
