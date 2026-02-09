project_root = "../../../.."
include(project_root.."/tools/build")

group("src")
project("xenia-ui-metal")
  uuid("f5a4b3c2-d1e8-4567-9abc-def123456789")
  language("C++")
  
  filter("system:macosx")
    kind("StaticLib")
  filter("not system:macosx")
    kind("None")
  filter({})
  
  links({
    "xenia-base",
    "xenia-ui",
    "metal-cpp",
  })
  includedirs({
    project_root.."/third_party/metal-cpp",
  })
  local_platform_files()
  files({
    "../shaders/bytecode/metal/*.h",
  })

  filter("system:macosx")
    links({
      "Metal.framework",
      "MetalFX.framework",
      "MetalKit.framework",
      "QuartzCore.framework",
    })
    files({
      "metal_immediate_drawer.mm",
      "metal_presenter.mm",
    })
  filter({})
