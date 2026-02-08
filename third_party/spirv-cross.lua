-- SPIRV-Cross premake5 build configuration for Xenia
-- Builds the core, GLSL, and MSL backends of SPIRV-Cross as a static library.

group("third_party")
project("spirv-cross")
  uuid("f1e2d3c4-b5a6-7890-fedc-ba9876543210")
  kind("StaticLib")
  language("C++")
  cppdialect("C++17")

  defines {
    -- Exceptions are enabled so SPIRV-Cross errors can be caught and handled
    -- gracefully instead of calling abort().  See msl_shader.cc CompileToMsl.
  }

  includedirs {
    "SPIRV-Cross",
  }

  files {
    -- Core
    "SPIRV-Cross/spirv_cross.cpp",
    "SPIRV-Cross/spirv_cross.hpp",
    "SPIRV-Cross/spirv_cross_parsed_ir.cpp",
    "SPIRV-Cross/spirv_cross_parsed_ir.hpp",
    "SPIRV-Cross/spirv_parser.cpp",
    "SPIRV-Cross/spirv_parser.hpp",
    "SPIRV-Cross/spirv_cfg.cpp",
    "SPIRV-Cross/spirv_cfg.hpp",
    "SPIRV-Cross/spirv_cross_util.cpp",
    "SPIRV-Cross/spirv_cross_util.hpp",
    "SPIRV-Cross/spirv_common.hpp",
    "SPIRV-Cross/spirv_cross_containers.hpp",
    "SPIRV-Cross/spirv_cross_error_handling.hpp",
    "SPIRV-Cross/spirv.hpp",
    -- GLSL backend (required by MSL backend)
    "SPIRV-Cross/spirv_glsl.cpp",
    "SPIRV-Cross/spirv_glsl.hpp",
    -- MSL backend
    "SPIRV-Cross/spirv_msl.cpp",
    "SPIRV-Cross/spirv_msl.hpp",
  }

  filter "system:macosx or ios"
    -- Enable on Apple platforms
  filter "not system:macosx"
    filter "not system:ios"
      -- Still build for potential cross-compilation tooling
    filter {}
  filter {}
