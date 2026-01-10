project_root = "../../.."
include(project_root.."/tools/build")

group("src")
project("xenia-hid")
  uuid("88a4ef38-c550-430f-8c22-8ded4e4ef601")
  kind("StaticLib")
  language("C++")
  links({
    "xenia-base",
  })
  defines({
  })
  local_platform_files()
  removefiles({"*_demo.cc"})

group("demos")
project("xenia-hid-demo")
  uuid("a56a209c-16d5-4913-85f9-86976fe7fddf")
  single_library_windowed_app_kind()
  language("C++")
  links({
    "fmt",
    "imgui",
    "xenia-base",
    "xenia-hid",
    "xenia-hid-nop",
    "xenia-ui",
  })
  filter("platforms:not Mac")
    links({
      "xenia-ui-vulkan",
    })
    includedirs({
      project_root.."/third_party/Vulkan-Headers/include",
    })
  filter({})
  files({
    "hid_demo.cc",
    "../ui/windowed_app_main_"..platform_suffix..".cc",
  })
  resincludedirs({
    project_root,
  })

  filter("platforms:not Android-*")
    links({
      "xenia-helper-sdl",
      "xenia-hid-sdl",
    })

  filter("system:macosx")
    links({
      "xenia-ui-metal",
      "metal-cpp",
      "SDL2",
      "Metal.framework",
      "MetalFX.framework",
      "MetalKit.framework",
      "QuartzCore.framework",
    })
    files({
      "Info.plist",
      project_root.."/xenia.entitlements",
    })
    buildoptions({
      "-DINFOPLIST_FILE=" .. path.getabsolute("Info.plist"),
    })
    xcodebuildsettings({
      ["INFOPLIST_FILE"] = path.getabsolute("Info.plist"),
      ["PRODUCT_BUNDLE_IDENTIFIER"] = "com.xenia.hid-demo",
      ["CODE_SIGN_STYLE"] = "Automatic",
      ["CODE_SIGN_ENTITLEMENTS"] =
          path.getabsolute(project_root.."/xenia.entitlements"),
    })

  filter("platforms:Linux")
    links({
      "SDL2",
      "X11",
      "xcb",
      "X11-xcb",
    })

  filter("platforms:Windows-*")
    links({
      "xenia-hid-winkey",
      "xenia-hid-xinput",
    })
