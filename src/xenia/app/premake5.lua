project_root = "../../.."
include(project_root.."/tools/build")

group("src")
project("xenia-app")
  uuid("d7e98620-d007-4ad8-9dbd-b47c8853a17f")
  language("C++")
  links({
    "xenia-apu",
    "xenia-apu-nop",
    "xenia-base",
    "xenia-core",
    "xenia-cpu",
    "xenia-gpu",
    "xenia-gpu-null",
    "xenia-hid",
    "xenia-hid-nop",
    "xenia-kernel",
    "xenia-ui",
    "xenia-vfs",
  })
  links({
    "aes_128",
    "capstone",
    "fmt",
    "dxbc",
    "discord-rpc",
    "imgui",
    "libavcodec",
    "libavutil",
    "mspack",
    "snappy",
    "xxhash",
  })
  filter("platforms:not Mac")
    links({
      "glslang-spirv",
      "xenia-gpu-vulkan",
      "xenia-ui-vulkan",
    })
  filter({})
  defines({
    "XBYAK_NO_OP_NAMES",
    "XBYAK_ENABLE_OMITTED_OPERAND",
  })
  local_platform_files()
  files({
    "../base/main_init_"..platform_suffix..".cc",
    "../ui/windowed_app_main_"..platform_suffix..".cc",
  })

  resincludedirs({
    project_root,
  })

  filter(SINGLE_LIBRARY_FILTER)
    -- Unified library containing all apps as StaticLibs, not just the main
    -- emulator windowed app.
    kind("SharedLib")
    links({
      "xenia-gpu-vulkan-trace-viewer",
      "xenia-hid-demo",
      "xenia-ui-window-vulkan-demo",
    })
  filter({"platforms:Mac", SINGLE_LIBRARY_FILTER})
    removelinks({
      "xenia-gpu-vulkan-trace-viewer",
      "xenia-ui-window-vulkan-demo",
    })
  filter(NOT_SINGLE_LIBRARY_FILTER)
    kind("WindowedApp")

  -- `targetname` is broken if building from Gradle, works only for toggling the
  -- `lib` prefix, as Gradle uses LOCAL_MODULE_FILENAME, not a derivative of
  -- LOCAL_MODULE, to specify the targets to build when executing ndk-build.
  filter("platforms:not Android-*")
    targetname("xenia")

  filter("architecture:x86_64")
    links({
      "xenia-cpu-backend-x64",
    })

  filter("architecture:ARM64")
    links({
      "xenia-cpu-backend-a64",
    })

  -- TODO(Triang3l): The emulator itself on Android.
  filter("platforms:not Android-*")
    files({
      "xenia_main.cc",
    })

  filter("platforms:Windows-*")
    files({
      "main_resources.rc",
    })

  filter({"platforms:Windows-*", "architecture:x86_64", "files:../base/main_init_"..platform_suffix..".cc"})
    -- MSVC x64 doesn't support /arch:IA32; remove AVX so the AVX check
    -- implementation itself doesn't get compiled with AVX enabled.
    removebuildoptions({
      "/arch:AVX",
      "/arch:AVX2",
    })

  filter("platforms:not Android-*")
    links({
      "xenia-apu-sdl",
      -- TODO(Triang3l): CPU debugger on Android.
      "xenia-debug-ui",
      "xenia-helper-sdl",
      "xenia-hid-sdl",
    })

  filter("platforms:Mac")
    local function select_libdir(candidates, filename)
      for _, dir in ipairs(candidates) do
        if os.isfile(path.join(dir, filename)) then
          return dir
        end
      end
      return nil
    end
    local metal_converter_libdir =
        path.getabsolute(path.join(project_root, "third_party/metal-shader-converter/lib"))
    local dxilconv_libdir =
        path.getabsolute(path.join(project_root, "third_party/DirectXShaderCompiler/build_dxilconv_macos/lib"))
    local lz4_libdir = "/opt/homebrew/opt/lz4/lib"
    local sdl2_libdir = "/opt/homebrew/opt/sdl2/lib"
    if os.istarget("macosx") then
      lz4_libdir = select_libdir({
        "/opt/homebrew/opt/lz4/lib",
        "/opt/homebrew/lib",
        "/usr/local/opt/lz4/lib",
        "/usr/local/lib",
      }, "liblz4.1.dylib") or lz4_libdir
      sdl2_libdir = select_libdir({
        "/opt/homebrew/opt/sdl2/lib",
        "/opt/homebrew/lib",
        "/usr/local/opt/sdl2/lib",
        "/usr/local/lib",
      }, "libSDL2-2.0.0.dylib") or sdl2_libdir
      if not os.isfile(path.join(lz4_libdir, "liblz4.1.dylib")) then
        error("LZ4 dylib not found. Install with `brew install lz4`.")
      end
      if not os.isfile(path.join(sdl2_libdir, "libSDL2-2.0.0.dylib")) then
        error("SDL2 dylib not found. Install with `brew install sdl2`.")
      end
    end
    -- Use the mac-specific windowed app entrypoint (avoid posix stub).
    removefiles({ "../ui/windowed_app_main_posix.cc" })
    files({ "../ui/windowed_app_main_mac.cc" })
    -- Disable Discord RPC on macOS (Windows-only binary).
    removelinks({
      "discord-rpc",
    })

  filter("platforms:Linux")
    links({
      "X11",
      "xcb",
      "X11-xcb",
      "SDL2",
    })

  filter("platforms:Windows-*")
    links({
      "xenia-app-discord",
      "xenia-apu-xaudio2",
      "xenia-gpu-d3d12",
      "xenia-hid-winkey",
      "xenia-hid-xinput",
      "xenia-ui-d3d12",
      -- Windows system libraries needed by dependencies.
      "dxguid",
      "ws2_32",
      -- SDL2 is built as a static library on Windows.
      "SDL2",
      "setupapi",
      "winmm",
      "imm32",
      "version",
    })

  filter({"platforms:Windows-*", SINGLE_LIBRARY_FILTER})
    links({
      "xenia-gpu-d3d12-trace-viewer",
      "xenia-ui-window-d3d12-demo",
    })

  filter("platforms:Windows-*")
    -- Only create the .user file if it doesn't already exist.
    local user_file = project_root.."/build/xenia-app.vcxproj.user"
    if not os.isfile(user_file) then
      debugdir(project_root)
      debugargs({
      })
    end

  filter("platforms:Mac")
    -- Link Metal UI/GPU on macOS so the Metal backend can be selected.
    local app_bundle = "${TARGET_BUILD_DIR}/${FULL_PRODUCT_NAME}"
    local app_contents = "${TARGET_BUILD_DIR}/${FULL_PRODUCT_NAME}/Contents"
    local app_frameworks = app_contents .. "/Frameworks"
    local app_executable = app_contents .. "/MacOS/xenia"
    local entitlements_path =
        path.getabsolute(project_root .. "/xenia.entitlements")
    links({
      "xenia-gpu-metal",
      "xenia-ui-metal",
      "metal-cpp",
      "metalirconverter",
      "dxilconv",
      "LLVMDxcSupport",
      "SDL2",
      "Metal.framework",
      "MetalKit.framework",
      "QuartzCore.framework",
    })
    libdirs({
      metal_converter_libdir,
      dxilconv_libdir,
      "/usr/local/lib",
    })
    runpathdirs({
      "@executable_path/../Frameworks",
      metal_converter_libdir,
      dxilconv_libdir,
      "/usr/local/lib",
    })
    linkoptions({
      "-Wl,-rpath,@executable_path/../Frameworks",
      "-Wl,-rpath,@loader_path/../Frameworks",
    })
    -- Bundle runtime dylibs inside the app bundle (Contents/Frameworks).
    postbuildcommands({
      'mkdir -p "' .. app_frameworks .. '"',
      'cp -f "' ..
          path.join(metal_converter_libdir, "libmetalirconverter.dylib") ..
          '" "' .. app_frameworks .. '/"',
      'cp -f "' ..
          path.join(dxilconv_libdir, "libdxilconv.dylib") ..
          '" "' .. app_frameworks .. '/"',
      'cp -f "' .. path.join(lz4_libdir, "liblz4.1.dylib") .. '" "' ..
          app_frameworks .. '/"',
      'cp -f "' .. path.join(sdl2_libdir, "libSDL2-2.0.0.dylib") .. '" "' ..
          app_frameworks .. '/"',
      'install_name_tool -id @rpath/liblz4.1.dylib "' .. app_frameworks ..
          '/liblz4.1.dylib"',
      'install_name_tool -id @rpath/libSDL2-2.0.0.dylib "' .. app_frameworks ..
          '/libSDL2-2.0.0.dylib"',
      'if otool -L "' .. app_executable .. '" | grep -q ' ..
          '"/opt/homebrew/opt/lz4/lib/liblz4.1.dylib"; then ' ..
          'install_name_tool -change ' ..
          '"/opt/homebrew/opt/lz4/lib/liblz4.1.dylib" ' ..
          '"@rpath/liblz4.1.dylib" "' .. app_executable .. '"; fi',
      'if otool -L "' .. app_executable .. '" | grep -q ' ..
          '"/usr/local/opt/lz4/lib/liblz4.1.dylib"; then ' ..
          'install_name_tool -change ' ..
          '"/usr/local/opt/lz4/lib/liblz4.1.dylib" ' ..
          '"@rpath/liblz4.1.dylib" "' .. app_executable .. '"; fi',
      'if otool -L "' .. app_executable .. '" | grep -q ' ..
          '"/opt/homebrew/opt/sdl2/lib/libSDL2-2.0.0.dylib"; then ' ..
          'install_name_tool -change ' ..
          '"/opt/homebrew/opt/sdl2/lib/libSDL2-2.0.0.dylib" ' ..
          '"@rpath/libSDL2-2.0.0.dylib" "' .. app_executable .. '"; fi',
      'if otool -L "' .. app_executable .. '" | grep -q ' ..
          '"/usr/local/opt/sdl2/lib/libSDL2-2.0.0.dylib"; then ' ..
          'install_name_tool -change ' ..
          '"/usr/local/opt/sdl2/lib/libSDL2-2.0.0.dylib" ' ..
          '"@rpath/libSDL2-2.0.0.dylib" "' .. app_executable .. '"; fi',
      'codesign --force --sign - "' .. app_frameworks ..
          '/libmetalirconverter.dylib"',
      'codesign --force --sign - "' .. app_frameworks ..
          '/libdxilconv.dylib"',
      'codesign --force --sign - "' .. app_frameworks ..
          '/liblz4.1.dylib"',
      'codesign --force --sign - "' .. app_frameworks ..
          '/libSDL2-2.0.0.dylib"',
      'codesign --force --deep --sign - --entitlements "' ..
          entitlements_path .. '" "' .. app_bundle .. '"',
    })
    files({
      "Info.plist",
      project_root.."/xenia.entitlements",
      project_root.."/assets/icon/xenia.icns",
    })
    filter({"platforms:Mac", "files:**.icns"})
      buildaction("Resources")
    filter("platforms:Mac")
    buildoptions({
      "-DINFOPLIST_FILE=" .. path.getabsolute("Info.plist"),
    })
    xcodebuildsettings({
      ["INFOPLIST_FILE"] = path.getabsolute("Info.plist"),
      ["PRODUCT_NAME"] = "Xenia",
      ["EXECUTABLE_NAME"] = "xenia",
      ["PRODUCT_BUNDLE_IDENTIFIER"] = "com.xenia.xenia",
      ["CODE_SIGN_STYLE"] = "Automatic",
      ["CODE_SIGN_ENTITLEMENTS"] = entitlements_path,
    })
