project_root = "../../../../../"
include(project_root.."/tools/build")

if os.istarget("macosx") then
  return
end

test_suite("xenia-gpu-shader-tests", project_root, ".", {
  links = {
    "fmt",
    "xenia-base",
    "xenia-gpu",
  },
  includedirs = {
    project_root.."/third_party/Vulkan-Headers/include",
    "/usr/include/vkd3d",  -- For vkd3d-shader DXBC conversion
  },
})

-- Use vkd3d-shader for DXBC conversion and SPIRV-Cross for reflection
filter("platforms:Linux-*")
  links({
    "vkd3d-shader",
    "spirv-cross-c-shared",  -- For SPIRV reflection (C API)
  })
filter({})

-- Add utility source files (test_suite only adds *_test.cc by default)
files({
  "util/vulkan_test_device.cc",
  "util/spirv_cross_wrapper.cc",
  "util/compute_test_harness.cc",
  "bytecode/*_test.cc",  -- Bytecode shader tests
})

-- Platform-specific Vulkan libraries
filter("platforms:Linux-*")
  links({
    "vulkan",  -- System Vulkan library
  })
filter("platforms:Windows-*")
  links({
    "vulkan-1",  -- Vulkan SDK library
  })
filter({})
