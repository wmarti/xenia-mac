/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/msl_shader.h"

#include <cstring>

#include "spirv_msl.hpp"

#include "xenia/base/assert.h"
#include "xenia/base/logging.h"

namespace xe {
namespace gpu {
namespace metal {

// SPIR-V descriptor set indices matching SpirvShaderTranslator's layout.
// These values must stay in sync with the enum in spirv_shader_translator.h.
namespace SpirvSets {
constexpr uint32_t kSharedMemoryAndEdram = 0;
constexpr uint32_t kConstants = 1;
constexpr uint32_t kTexturesVertex = 2;
constexpr uint32_t kTexturesPixel = 3;
}  // namespace SpirvSets

// SPIR-V constant buffer binding indices within descriptor set 1.
// Must match SpirvShaderTranslator::ConstantBuffer enum.
namespace SpirvCbv {
constexpr uint32_t kSystem = 0;
constexpr uint32_t kFloatVertex = 1;
constexpr uint32_t kFloatPixel = 2;
constexpr uint32_t kBoolLoop = 3;
constexpr uint32_t kFetch = 4;
constexpr uint32_t kClipPlanes = 5;
constexpr uint32_t kTessellation = 6;
}  // namespace SpirvCbv

MslShader::MslShader(xenos::ShaderType shader_type, uint64_t ucode_data_hash,
                     const uint32_t* ucode_dwords, size_t ucode_dword_count,
                     std::endian ucode_source_endian)
    : SpirvShader(shader_type, ucode_data_hash, ucode_dwords, ucode_dword_count,
                  ucode_source_endian) {}

Shader::Translation* MslShader::CreateTranslationInstance(
    uint64_t modification) {
  return new MslTranslation(*this, modification);
}

MslShader::MslTranslation::~MslTranslation() {
  if (metal_function_) {
    metal_function_->release();
    metal_function_ = nullptr;
  }
  if (metal_library_) {
    metal_library_->release();
    metal_library_ = nullptr;
  }
}

// Metal buffer binding indices for resources.
// These must match the SPIRV-Cross remapping done below.
namespace MslBindings {
// Descriptor Set 0: Shared memory and EDRAM.
constexpr uint32_t kSharedMemory = 0;
// Descriptor Set 1: Constant buffers.
constexpr uint32_t kSystemConstants = 1;
constexpr uint32_t kFloatConstantsVertex = 2;
constexpr uint32_t kFloatConstantsPixel = 3;
constexpr uint32_t kBoolLoopConstants = 4;
constexpr uint32_t kFetchConstants = 5;
constexpr uint32_t kClipPlaneConstants = 6;
constexpr uint32_t kTessellationConstants = 7;
// Descriptor Set 2/3: Textures start at this index.
constexpr uint32_t kTextureBase = 0;
// Samplers start at this index.
constexpr uint32_t kSamplerBase = 0;
}  // namespace MslBindings

static void AddResourceBindings(spirv_cross::CompilerMSL& compiler,
                                spv::ExecutionModel stage) {
  using MSLBinding = spirv_cross::MSLResourceBinding;

  // Set 0: Shared memory (storage buffer at binding 0).
  {
    MSLBinding binding;
    binding.stage = stage;
    binding.desc_set = SpirvSets::kSharedMemoryAndEdram;
    binding.binding = 0;
    binding.msl_buffer = MslBindings::kSharedMemory;
    binding.msl_texture = 0;
    binding.msl_sampler = 0;
    compiler.add_msl_resource_binding(binding);
  }

  // Set 0 binding 1: EDRAM (only used in FSI mode — skip for host RT path).
  // We still add a dummy mapping so SPIRV-Cross doesn't complain if the
  // shader happens to reference it (it shouldn't for kHostRenderTargets).
  {
    MSLBinding binding;
    binding.stage = stage;
    binding.desc_set = SpirvSets::kSharedMemoryAndEdram;
    binding.binding = 1;
    binding.msl_buffer = 30;  // High index, unused.
    binding.msl_texture = 0;
    binding.msl_sampler = 0;
    compiler.add_msl_resource_binding(binding);
  }

  // Set 1: Constant buffers.
  struct CbvMapping {
    uint32_t spirv_binding;
    uint32_t msl_buffer;
  };
  static const CbvMapping cbv_mappings[] = {
      {SpirvCbv::kSystem, MslBindings::kSystemConstants},
      {SpirvCbv::kFloatVertex, MslBindings::kFloatConstantsVertex},
      {SpirvCbv::kFloatPixel, MslBindings::kFloatConstantsPixel},
      {SpirvCbv::kBoolLoop, MslBindings::kBoolLoopConstants},
      {SpirvCbv::kFetch, MslBindings::kFetchConstants},
      {SpirvCbv::kClipPlanes, MslBindings::kClipPlaneConstants},
      {SpirvCbv::kTessellation, MslBindings::kTessellationConstants},
  };
  for (const auto& cbv : cbv_mappings) {
    MSLBinding binding;
    binding.stage = stage;
    binding.desc_set = SpirvSets::kConstants;
    binding.binding = cbv.spirv_binding;
    binding.msl_buffer = cbv.msl_buffer;
    binding.msl_texture = 0;
    binding.msl_sampler = 0;
    compiler.add_msl_resource_binding(binding);
  }

  // Set 2 (vertex textures) and Set 3 (pixel textures): up to 32 textures.
  // Map all possible texture bindings.
  for (uint32_t set = SpirvSets::kTexturesVertex;
       set <= SpirvSets::kTexturesPixel; ++set) {
    for (uint32_t i = 0; i < 32; ++i) {
      // Texture binding.
      {
        MSLBinding binding;
        binding.stage = stage;
        binding.desc_set = set;
        binding.binding = i;
        binding.msl_buffer = 0;
        binding.msl_texture = MslBindings::kTextureBase + i;
        binding.msl_sampler = 0;
        compiler.add_msl_resource_binding(binding);
      }
      // Sampler binding (same index).
      {
        MSLBinding binding;
        binding.stage = stage;
        binding.desc_set = set;
        binding.binding = i;
        binding.msl_buffer = 0;
        binding.msl_texture = 0;
        binding.msl_sampler = MslBindings::kSamplerBase + i;
        compiler.add_msl_resource_binding(binding);
      }
    }
  }
}

bool MslShader::MslTranslation::CompileToMsl(MTL::Device* device, bool is_ios) {
  if (!device) {
    XELOGE("MslShader: No Metal device provided");
    return false;
  }

  const std::vector<uint8_t>& spirv_data = translated_binary();
  if (spirv_data.empty()) {
    XELOGE("MslShader: No translated SPIR-V data available");
    return false;
  }

  // SPIR-V data is a vector of bytes, but SPIRV-Cross expects uint32_t words.
  if (spirv_data.size() % sizeof(uint32_t) != 0) {
    XELOGE("MslShader: SPIR-V data size is not aligned to 4 bytes");
    return false;
  }
  const uint32_t* spirv_words =
      reinterpret_cast<const uint32_t*>(spirv_data.data());
  size_t spirv_word_count = spirv_data.size() / sizeof(uint32_t);

  try {
    spirv_cross::CompilerMSL compiler(spirv_words, spirv_word_count);

    // Configure MSL options.
    auto opts = compiler.get_msl_options();
    if (is_ios) {
      opts.platform = spirv_cross::CompilerMSL::Options::iOS;
    } else {
      opts.platform = spirv_cross::CompilerMSL::Options::macOS;
    }
    // MSL 2.4 (macOS 12+ / iOS 15+) — supports argument buffers,
    // simdgroup functions, raster order groups.
    opts.msl_version =
        spirv_cross::CompilerMSL::Options::make_msl_version(2, 4);
    // Use direct buffer/texture/sampler bindings (no argument buffers).
    // This is simpler and avoids the indirection overhead of the old
    // IRDescriptorTable model. Can be switched to argument buffers later
    // if CPU binding overhead becomes a bottleneck.
    opts.argument_buffers = false;
    // Use simdgroup functions on iOS (A13+).
    opts.ios_use_simdgroup_functions = true;
    // Force sample rate shading to be available if needed.
    opts.force_sample_rate_shading = false;
    // Ensure buffer sizes are not padded.
    opts.pad_fragment_output_components = false;
    compiler.set_msl_options(opts);

    // Remap SPIR-V descriptor sets/bindings to Metal buffer/texture/sampler
    // indices.
    spv::ExecutionModel execution_model =
        shader().type() == xenos::ShaderType::kVertex
            ? spv::ExecutionModelVertex
            : spv::ExecutionModelFragment;
    AddResourceBindings(compiler, execution_model);

    // Compile to MSL.
    msl_source_ = compiler.compile();
    if (msl_source_.empty()) {
      XELOGE("MslShader: SPIRV-Cross compilation produced empty output");
      return false;
    }

    // Get the entry point name that SPIRV-Cross chose.
    entry_point_name_ =
        compiler.get_cleansed_entry_point_name("main", execution_model);
    if (entry_point_name_.empty()) {
      // Fallback — SPIRV-Cross often names it "main0".
      entry_point_name_ = "main0";
    }

    XELOGD("MslShader: Compiled SPIR-V to MSL ({} bytes, entry: {})",
           msl_source_.size(), entry_point_name_);

  } catch (const spirv_cross::CompilerError& e) {
    XELOGE("MslShader: SPIRV-Cross compilation failed: {}", e.what());
    return false;
  }

  // Compile MSL source to a Metal library.
  NS::Error* error = nullptr;
  auto* source_str =
      NS::String::string(msl_source_.c_str(), NS::UTF8StringEncoding);
  auto* compile_options = MTL::CompileOptions::alloc()->init();
  // Use fast math for better performance (matches Xbox 360 behavior better
  // than strict IEEE — the 360's ALU doesn't fully conform to IEEE anyway).
  compile_options->setFastMathEnabled(true);

  metal_library_ = device->newLibrary(source_str, compile_options, &error);
  compile_options->release();

  if (!metal_library_) {
    if (error) {
      XELOGE("MslShader: Metal library compilation failed: {}",
             error->localizedDescription()->utf8String());
    } else {
      XELOGE("MslShader: Metal library compilation failed (unknown error)");
    }
    return false;
  }

  // Get the entry point function.
  auto* fn_name =
      NS::String::string(entry_point_name_.c_str(), NS::UTF8StringEncoding);
  metal_function_ = metal_library_->newFunction(fn_name);
  if (!metal_function_) {
    XELOGE("MslShader: Could not find function '{}' in compiled library",
           entry_point_name_);
    metal_library_->release();
    metal_library_ = nullptr;
    return false;
  }

  XELOGD("MslShader: Successfully created Metal function '{}'",
         entry_point_name_);
  return true;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
