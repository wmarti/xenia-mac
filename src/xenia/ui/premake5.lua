project_root = "../../.."
include(project_root.."/tools/build")

group("src")
project("xenia-ui")
  uuid("d0407c25-b0ea-40dc-846c-82c46fbd9fa2")
  kind("StaticLib")
  language("C++")
  links({
    "xenia-base",
  })
  defines({
  })
  local_platform_files()
  removefiles({"*_demo.cc"})
  removefiles({"windowed_app_main_*.cc"})

  filter("platforms:Android-*")
    -- Exports JNI functions.
    wholelib("On")

  filter("platforms:Linux-*")
    -- Use pkg-config to get GTK include paths and libraries
    buildoptions({
      "`pkg-config --cflags gtk+-3.0`",
    })
    linkoptions({
      "`pkg-config --libs gtk+-3.0`",
    })

  filter("platforms:Windows-*")
    links({
      "dxgi",
      "dwmapi",
      "shlwapi",  -- For QISearch and other shell functions
    })
