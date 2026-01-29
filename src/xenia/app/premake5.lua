project_root = path.getabsolute("../../..")
include(project_root.."/tools/build")

-- macOS library paths for Metal backend dependencies.
local metal_converter_libdir =
    path.join(project_root, "third_party/metal-shader-converter/lib")
local dxilconv_libdir_arm64 =
    path.join(project_root,
              "third_party/DirectXShaderCompiler/build_dxilconv_macos/lib")
local dxilconv_libdir_x86_64 =
    path.join(project_root,
              "third_party/DirectXShaderCompiler/build_dxilconv_macos_x86_64/lib")

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
    "xenia-gpu-vulkan",
    "xenia-hid",
    "xenia-hid-nop",
    "xenia-kernel",
    "xenia-patcher",
    "xenia-ui",
    "xenia-ui-vulkan",
    "xenia-vfs",
  })
  links({
    "aes_128",
    "capstone",
    "fmt",
    "dxbc",
    "discord-rpc",
    "glslang-spirv",
    "imgui",
    "libavcodec",
    "libavutil",
    "mspack",
    "snappy",
    "xxhash",
  })
  defines({
    "XBYAK_NO_OP_NAMES",
    "XBYAK_ENABLE_OMITTED_OPERAND",
  })
  apu_transitive_deps()
  local_platform_files()
  files({
    "../base/main_init_"..platform_suffix..".cc",
    "../ui/windowed_app_main_qt.cc",
  })

  resincludedirs({
    project_root,
  })

  filter(SINGLE_LIBRARY_FILTER)
    -- Unified library containing all apps as StaticLibs, not just the main
    -- emulator windowed app.
    kind("SharedLib")
  if enableMiscSubprojects then
      links({
        "xenia-gpu-vulkan-trace-viewer",
        "xenia-hid-demo",
        "xenia-ui-window-vulkan-demo",
      })
  end
  filter(NOT_SINGLE_LIBRARY_FILTER)
    kind("WindowedApp")
  filter({NOT_SINGLE_LIBRARY_FILTER, "platforms:Windows-*", "configurations:Debug"})
    kind("ConsoleApp")

  -- `targetname` is broken if building from Gradle, works only for toggling the
  -- `lib` prefix, as Gradle uses LOCAL_MODULE_FILENAME, not a derivative of
  -- LOCAL_MODULE, to specify the targets to build when executing ndk-build.
  filter("platforms:Mac-*")
    targetname("xenia")
  filter({"platforms:not Android-*", "platforms:not Mac-*"})
    targetname("xenia_edge")

  filter("architecture:x86_64")
    links({
      "xenia-cpu-backend-x64",
    })
  filter("architecture:arm64")
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
    linkoptions({"/ENTRY:mainCRTStartup"})

  filter({"architecture:x86_64", "files:../base/main_init_"..platform_suffix..".cc"})
    vectorextensions("SSE2")  -- Disable AVX for main_init_win.cc so our AVX check doesn't use AVX instructions.

  filter("platforms:not Android-*")
    links({
      "xenia-app-discord",
      "xenia-apu-sdl",
      -- TODO(Triang3l): CPU debugger on Android.
      "xenia-debug-ui",
      "xenia-helper-sdl",
      "xenia-hid-sdl",
    })

  filter("platforms:Linux-*")
    links({
      "xenia-apu-alsa",
      "X11",
      "xcb",
      "X11-xcb",
      "SDL2",
    })

  -- macOS: Use Metal backend instead of Vulkan.
  filter("platforms:Mac-*")
    removelinks({
      "xenia-gpu-vulkan",
      "xenia-ui-vulkan",
    })
    links({
      "xenia-gpu-metal",
      "xenia-ui-metal",
      "metal-cpp",
      "metalirconverter",
      "dxilconv",
      "LLVMDxcSupport",
      "SDL2",
      "Cocoa.framework",
      "CoreFoundation.framework",
      "Metal.framework",
      "MetalFX.framework",
      "MetalKit.framework",
      "QuartzCore.framework",
    })
    libdirs({
      metal_converter_libdir,
    })
    linkoptions({
      "-Wl,-rpath,@executable_path/../Frameworks",
    })
  filter({"platforms:Mac-*", "architecture:arm64"})
    libdirs({ dxilconv_libdir_arm64, "/opt/homebrew/lib" })
    runpathdirs({ dxilconv_libdir_arm64, "/opt/homebrew/lib" })
    linkoptions({
      path.getabsolute(path.join(dxilconv_libdir_arm64, "libdxilconv.dylib")),
    })
    -- Copy dylibs to app bundle Frameworks folder
    local app_frameworks = "${TARGET_BUILD_DIR}/${FULL_PRODUCT_NAME}/Contents/Frameworks"
    local app_executable = "${TARGET_BUILD_DIR}/${FULL_PRODUCT_NAME}/Contents/MacOS/xenia"
    postbuildcommands({
      'mkdir -p "' .. app_frameworks .. '"',
      'cp -f "' .. path.getabsolute(path.join(metal_converter_libdir, "libmetalirconverter.dylib")) .. '" "' .. app_frameworks .. '/"',
      'cp -f "' .. path.getabsolute(path.join(dxilconv_libdir_arm64, "libdxilconv.dylib")) .. '" "' .. app_frameworks .. '/"',
      'codesign --force --sign - "' .. app_frameworks .. '/libmetalirconverter.dylib"',
      'codesign --force --sign - "' .. app_frameworks .. '/libdxilconv.dylib"',
    })
  filter({"platforms:Mac-*", "architecture:x86_64"})
    libdirs({ dxilconv_libdir_x86_64, "/usr/local/lib" })
    runpathdirs({ dxilconv_libdir_x86_64, "/usr/local/lib" })
    removelinks({ "LLVMDxcSupport" })
    linkoptions({
      path.getabsolute(path.join(dxilconv_libdir_x86_64, "libdxilconv.dylib")),
      path.getabsolute(path.join(dxilconv_libdir_x86_64, "libLLVMDxcSupport.a")),
    })
    -- Copy dylibs to app bundle Frameworks folder
    local app_frameworks_x86 = "${TARGET_BUILD_DIR}/${FULL_PRODUCT_NAME}/Contents/Frameworks"
    postbuildcommands({
      'mkdir -p "' .. app_frameworks_x86 .. '"',
      'cp -f "' .. path.getabsolute(path.join(metal_converter_libdir, "libmetalirconverter.dylib")) .. '" "' .. app_frameworks_x86 .. '/"',
      'cp -f "' .. path.getabsolute(path.join(dxilconv_libdir_x86_64, "libdxilconv.dylib")) .. '" "' .. app_frameworks_x86 .. '/"',
      'codesign --force --sign - "' .. app_frameworks_x86 .. '/libmetalirconverter.dylib"',
      'codesign --force --sign - "' .. app_frameworks_x86 .. '/libdxilconv.dylib"',
    })
  filter({})

  filter("platforms:Windows-*")
    links({
      "xenia-apu-xaudio2",
      "xenia-gpu-d3d12",
      "xenia-hid-winkey",
      "xenia-hid-xinput",
      "xenia-ui-d3d12",
    })

  filter("platforms:Windows-*")

  if enableMiscSubprojects then
    filter({"platforms:Windows-*", SINGLE_LIBRARY_FILTER})
      links({
        "xenia-gpu-d3d12-trace-viewer",
        "xenia-ui-window-d3d12-demo",
      })
  end

  filter("platforms:Windows-*")
    -- Only create the .user file if it doesn't already exist.
    local user_file = project_root.."/build/xenia-app.vcxproj.user"
    if not os.isfile(user_file) then
      debugdir(project_root)
    end

  -- Run windeployqt as post-build event to copy Qt DLLs
  filter({"platforms:Windows-*", "configurations:Debug or Checked"})
    local qt_dir = os.getenv("QT_DIR")
    if qt_dir then
      local windeployqt = path.translate(path.join(qt_dir, "bin", "windeployqt.exe"), "\\")
      postbuildcommands {
        'if exist "' .. windeployqt .. '" "' .. windeployqt .. '" --debug --no-translations --no-system-d3d-compiler --no-opengl-sw --no-compiler-runtime "$(TargetPath)"'
      }
    end

  filter({"platforms:Windows-*", "configurations:Release"})
    local qt_dir = os.getenv("QT_DIR")
    if qt_dir then
      local windeployqt = path.translate(path.join(qt_dir, "bin", "windeployqt.exe"), "\\")
      postbuildcommands {
        'if exist "' .. windeployqt .. '" "' .. windeployqt .. '" --release --no-translations --no-system-d3d-compiler --no-opengl-sw --no-compiler-runtime "$(TargetPath)"'
      }
    end

  -- Copy optimized-settings JSON files next to executable
  filter("platforms:Windows-*")
    -- Use absolute path to avoid issues with relative paths
    local optimized_settings_src = path.translate(path.getabsolute(path.join(project_root, ".data_repos", "optimized-settings", "settings")), "\\")
    postbuildcommands {
      'if not exist "$(TargetDir)optimized_settings" mkdir "$(TargetDir)optimized_settings"',
      'xcopy /I /Y /Q "' .. optimized_settings_src .. '\\*.json" "$(TargetDir)optimized_settings\\"'
    }

  -- Copy game-patches TOML files next to executable
  filter("platforms:Windows-*")
    local game_patches_src = path.translate(path.getabsolute(path.join(project_root, ".data_repos", "game-patches", "patches")), "\\")
    postbuildcommands {
      'if not exist "$(TargetDir)game_patches" mkdir "$(TargetDir)game_patches"',
      'xcopy /I /Y /Q "' .. game_patches_src .. '\\*.toml" "$(TargetDir)game_patches\\"'
    }

  -- Copy assets/font next to executable
  filter("platforms:Windows-*")
    local assets_font_src = path.translate(path.getabsolute(path.join(project_root, "assets", "font")), "\\")
    postbuildcommands {
      'if not exist "$(TargetDir)assets\\font" mkdir "$(TargetDir)assets\\font"',
      'xcopy /I /Y /Q "' .. assets_font_src .. '\\*.*" "$(TargetDir)assets\\font\\"'
    }

  filter("platforms:Linux-*")
    local optimized_settings_src = path.getabsolute(path.join(project_root, ".data_repos", "optimized-settings", "settings"))
    local optimized_settings_dst = path.getabsolute(path.join(project_root, "build", "bin", "Linux")) .. "/%{cfg.buildcfg}/optimized_settings"
    postbuildcommands {
      '{MKDIR} ' .. optimized_settings_dst,
      '{COPY} ' .. optimized_settings_src .. '/*.json ' .. optimized_settings_dst
    }

  filter("platforms:Linux-*")
    local game_patches_src = path.getabsolute(path.join(project_root, ".data_repos", "game-patches", "patches"))
    local game_patches_dst = path.getabsolute(path.join(project_root, "build", "bin", "Linux")) .. "/%{cfg.buildcfg}/game_patches"
    postbuildcommands {
      '{MKDIR} ' .. game_patches_dst,
      '{COPY} ' .. game_patches_src .. '/*.toml ' .. game_patches_dst
    }

  -- Copy assets/font next to executable
  filter("platforms:Linux-*")
    local assets_font_src = path.getabsolute(path.join(project_root, "assets", "font"))
    local assets_font_dst = path.getabsolute(path.join(project_root, "build", "bin", "Linux")) .. "/%{cfg.buildcfg}/assets/font"
    postbuildcommands {
      '{MKDIR} ' .. assets_font_dst,
      '{COPY} ' .. assets_font_src .. '/* ' .. assets_font_dst
    }

  -- macOS app bundle configuration.
  filter("platforms:Mac-*")
    local entitlements_path = path.getabsolute(project_root .. "/xenia.entitlements")
    local app_bundle = "${TARGET_BUILD_DIR}/${FULL_PRODUCT_NAME}"
    local app_contents = app_bundle .. "/Contents"
    local app_frameworks = app_contents .. "/Frameworks"
    local app_executable = app_contents .. "/MacOS/xenia"
    files({
      "Info.plist",
      project_root.."/xenia.entitlements",
      project_root.."/assets/icon/xenia.icns",
    })
    xcodebuildsettings({
      ["INFOPLIST_FILE"] = path.getabsolute("Info.plist"),
      ["MACOSX_DEPLOYMENT_TARGET"] = "15.0",
      ["PRODUCT_NAME"] = "Xenia",
      ["EXECUTABLE_NAME"] = "xenia",
      ["PRODUCT_BUNDLE_IDENTIFIER"] = "com.xenia.xenia-edge",
      ["CODE_SIGN_STYLE"] = "Automatic",
      ["CODE_SIGN_ENTITLEMENTS"] = entitlements_path,
      ["CODE_SIGN_ALLOW_ENTITLEMENTS_MODIFICATION"] = "YES",
    })
    postbuildcommands({
      'mkdir -p "' .. app_frameworks .. '"',
      'cp -f "' .. path.join(metal_converter_libdir, "libmetalirconverter.dylib")
          .. '" "' .. app_frameworks .. '/"',
      'codesign --force --sign - "' .. app_frameworks
          .. '/libmetalirconverter.dylib"',
      'codesign --force --deep --sign - --entitlements "'
          .. entitlements_path .. '" "' .. app_bundle .. '"',
    })
  filter({"platforms:Mac-*", "architecture:arm64"})
    postbuildcommands({
      'cp -f "' .. path.join(dxilconv_libdir_arm64, "libdxilconv.dylib")
          .. '" "' .. app_frameworks .. '/"',
      'codesign --force --sign - "' .. app_frameworks .. '/libdxilconv.dylib"',
    })
  filter({"platforms:Mac-*", "architecture:x86_64"})
    postbuildcommands({
      'cp -f "' .. path.join(dxilconv_libdir_x86_64, "libdxilconv.dylib")
          .. '" "' .. app_frameworks .. '/"',
      'codesign --force --sign - "' .. app_frameworks .. '/libdxilconv.dylib"',
    })
  filter({"platforms:Mac-*", "files:**.icns"})
    buildaction("Resources")
