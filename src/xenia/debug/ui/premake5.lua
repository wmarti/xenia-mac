project_root = "../../../.."
include(project_root.."/tools/build")

if TARGET_ARCH ~= "x86_64" then
  return
end

group("src")
project("xenia-debug-ui")
  uuid("9193a274-f4c2-4746-bd85-93fcfc5c3e38")
  kind("StaticLib")
  language("C++")
  links({
    "imgui",
    "xenia-base",
    "xenia-cpu",
    "xenia-ui",
  })
  local_platform_files()
