/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/msl_shader.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "spirv_msl.hpp"

#include "xenia/base/assert.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/metal/msl_bindings.h"

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

static const char* GetExecutionModelName(spv::ExecutionModel stage) {
  switch (stage) {
    case spv::ExecutionModelVertex:
      return "vertex";
    case spv::ExecutionModelFragment:
      return "fragment";
    case spv::ExecutionModelTessellationControl:
      return "tess_control";
    case spv::ExecutionModelTessellationEvaluation:
      return "tess_evaluation";
    case spv::ExecutionModelGLCompute:
      return "compute";
    default:
      return "other";
  }
}

static void AddResourceBindings(spirv_cross::CompilerMSL& compiler,
                                spv::ExecutionModel stage,
                                uint64_t shader_hash) {
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

  // Set 2 (vertex textures) and Set 3 (pixel textures): texture bindings.
  //
  // The SPIR-V translator places sampler bindings AFTER texture bindings in
  // the same descriptor set (sampler i gets SPIR-V binding texture_count + i).
  // Metal only supports 16 samplers per stage, so we must remap sampler
  // bindings to compact indices 0..M-1 regardless of their SPIR-V binding.
  //
  // Use a single MSLResourceBinding entry per SPIR-V binding to avoid
  // overwrite issues (SPIRV-Cross keys entries by {stage, set, binding}).
  //
  // Step 1: Map all 32 possible texture bindings with identity mapping.
  //         Set msl_sampler = 0 as a safe default (overridden for real
  //         samplers in step 2).
  for (uint32_t set = SpirvSets::kTexturesVertex;
       set <= SpirvSets::kTexturesPixel; ++set) {
    for (uint32_t i = 0; i < MslTextureIndex::kMaxPerStage; ++i) {
      MSLBinding binding;
      binding.stage = stage;
      binding.desc_set = set;
      binding.binding = i;
      binding.msl_buffer = 0;
      binding.msl_texture = MslBindings::kTextureBase + i;
      binding.msl_sampler = 0;
      compiler.add_msl_resource_binding(binding);
    }
  }

  // Step 2: Query the actual SPIR-V resources to find sampler bindings
  //         and remap them to compact Metal sampler indices 0..M-1.
  //         This handles the case where SPIR-V sampler bindings exceed 15
  //         (which would violate Metal's 16-sampler-per-stage limit).
  auto resources = compiler.get_shader_resources();
  struct SamplerSpvBinding {
    uint32_t set;
    uint32_t binding;
  };
  std::vector<SamplerSpvBinding> sampler_spv_bindings;
  sampler_spv_bindings.reserve(resources.separate_samplers.size());
  for (const auto& samp : resources.separate_samplers) {
    uint32_t set =
        compiler.get_decoration(samp.id, spv::DecorationDescriptorSet);
    uint32_t spv_binding =
        compiler.get_decoration(samp.id, spv::DecorationBinding);
    sampler_spv_bindings.push_back({set, spv_binding});
  }
  bool sampler_order_was_unsorted = false;
  for (size_t i = 1; i < sampler_spv_bindings.size(); ++i) {
    const auto& prev = sampler_spv_bindings[i - 1];
    const auto& curr = sampler_spv_bindings[i];
    if (curr.set < prev.set ||
        (curr.set == prev.set && curr.binding < prev.binding)) {
      sampler_order_was_unsorted = true;
      break;
    }
  }
  std::sort(sampler_spv_bindings.begin(), sampler_spv_bindings.end(),
            [](const SamplerSpvBinding& a, const SamplerSpvBinding& b) {
              if (a.set != b.set) {
                return a.set < b.set;
              }
              return a.binding < b.binding;
            });
  if (sampler_order_was_unsorted) {
    XELOGD(
        "MslShader: Reordered sampler SPIR-V bindings by set/binding for "
        "stable Metal remap shader={:016X} stage={}",
        shader_hash, GetExecutionModelName(stage));
  }
  uint32_t sampler_msl_index = 0;
  std::ostringstream sampler_remap_log;
  uint32_t sampler_remap_count = 0;
  for (size_t i = 0; i < sampler_spv_bindings.size(); ++i) {
    uint32_t set = sampler_spv_bindings[i].set;
    uint32_t spv_binding = sampler_spv_bindings[i].binding;
    if (i > 0 && set == sampler_spv_bindings[i - 1].set &&
        spv_binding == sampler_spv_bindings[i - 1].binding) {
      XELOGW(
          "MslShader: Duplicate sampler SPIR-V binding at set={} binding={} "
          "shader={:016X}",
          set, spv_binding, shader_hash);
      continue;
    }
    if (sampler_msl_index >= 16) {
      XELOGW(
          "MslShader: Too many samplers ({} >= 16), sampler at set={} "
          "binding={} will not be bound",
          sampler_msl_index, set, spv_binding);
      break;
    }
    // Overwrite the entry for this binding with the compact sampler index.
    // For standalone samplers, don't propagate the SPIR-V binding to
    // msl_texture; sampler SPIR-V bindings are after image bindings and may
    // exceed the per-stage Metal texture slot limit.
    MSLBinding binding;
    binding.stage = stage;
    binding.desc_set = set;
    binding.binding = spv_binding;
    binding.msl_buffer = 0;
    binding.msl_texture = MslTextureIndex::kBase;
    binding.msl_sampler = MslBindings::kSamplerBase + sampler_msl_index;
    compiler.add_msl_resource_binding(binding);
    sampler_remap_log << "[set=" << set << ",binding=" << spv_binding
                      << "->msl_sampler=" << sampler_msl_index << "] ";
    sampler_msl_index++;
    sampler_remap_count++;
  }
  if (sampler_remap_count) {
    XELOGD(
        "MslShader: Sampler remap table shader={:016X} stage={} count={} {}",
        shader_hash, GetExecutionModelName(stage), sampler_remap_count,
        sampler_remap_log.str());
  } else {
    XELOGD("MslShader: Sampler remap table shader={:016X} stage={} count=0",
           shader_hash, GetExecutionModelName(stage));
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

  // SPIRV-Cross throws exceptions on translation errors.  We catch them
  // below and return false so the draw can be skipped gracefully instead
  // of crashing the process.
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
  opts.msl_version = spirv_cross::CompilerMSL::Options::make_msl_version(2, 4);
  // Use direct buffer/texture/sampler bindings (no argument buffers).
  // This is simpler and avoids the indirection overhead of the old
  // IRDescriptorTable model. Can be switched to argument buffers later
  // if CPU binding overhead becomes a bottleneck.
  opts.argument_buffers = false;
  // Use simdgroup functions on iOS (A13+).
  opts.ios_use_simdgroup_functions = is_ios;
  // Force sample rate shading to be available if needed.
  opts.force_sample_rate_shading = false;
  // Ensure buffer sizes are not padded.
  opts.pad_fragment_output_components = false;
  compiler.set_msl_options(opts);

  // Remap SPIR-V descriptor sets/bindings to Metal buffer/texture/sampler
  // indices.
  // Query the actual execution model from the SPIR-V binary rather than
  // assuming kVertex always maps to spv::ExecutionModelVertex.  The
  // SpirvShaderTranslator already sets the correct model — domain shaders
  // (tessellation evaluation) are emitted as TessellationEvaluation, not
  // Vertex — so we must honour that for SPIRV-Cross resource remapping and
  // entry-point lookup.
  spv::ExecutionModel execution_model;
  {
    auto entry_points = compiler.get_entry_points_and_stages();
    if (!entry_points.empty()) {
      execution_model = entry_points[0].execution_model;
    } else {
      // Fallback: infer from shader type (pre-tessellation behavior).
      execution_model = shader().type() == xenos::ShaderType::kVertex
                            ? spv::ExecutionModelVertex
                            : spv::ExecutionModelFragment;
    }
  }
  AddResourceBindings(compiler, execution_model, shader().ucode_data_hash());

  // Validate resource counts against Metal limits before compilation.
  {
    auto resources = compiler.get_shader_resources();
    bool expects_tessellation_constants = false;
    for (const auto& uniform_buffer : resources.uniform_buffers) {
      uint32_t set = compiler.get_decoration(
          uniform_buffer.id, spv::DecorationDescriptorSet);
      uint32_t binding =
          compiler.get_decoration(uniform_buffer.id, spv::DecorationBinding);
      if (set == SpirvSets::kConstants && binding == SpirvCbv::kTessellation) {
        expects_tessellation_constants = true;
        break;
      }
    }
    size_t texture_count =
        resources.sampled_images.size() + resources.separate_images.size();
    size_t sampler_count = resources.separate_samplers.size();
    XELOGD(
        "MslShader: Resource summary shader={:016X} stage={} textures={} "
        "samplers={} uniform_buffers={} tessellation_cbv_expected={}",
        shader().ucode_data_hash(), GetExecutionModelName(execution_model),
        texture_count, sampler_count, resources.uniform_buffers.size(),
        expects_tessellation_constants ? 1 : 0);
    if (expects_tessellation_constants) {
      XELOGW(
          "MslShader: Shader {:016X} stage={} declares set={},binding={} "
          "tessellation constants (msl_buffer={})",
          shader().ucode_data_hash(), GetExecutionModelName(execution_model),
          SpirvSets::kConstants, SpirvCbv::kTessellation,
          MslBindings::kTessellationConstants);
    }
    if (texture_count > MslTextureIndex::kMaxPerStage) {
      XELOGE(
          "MslShader: Shader uses {} textures, exceeding Metal's {}-per-stage "
          "limit in this backend",
          texture_count, MslTextureIndex::kMaxPerStage);
      return false;
    }
    if (sampler_count > 16) {
      XELOGE(
          "MslShader: Shader uses {} samplers, exceeding Metal's 16-per-stage "
          "limit — translation should be considered failed",
          sampler_count);
      return false;
    }
  }

  // Compile to MSL.  SPIRV-Cross throws on translation errors.
  try {
    msl_source_ = compiler.compile();
  } catch (const std::exception& e) {
    XELOGE("MslShader: SPIRV-Cross compilation failed: {}", e.what());
    return false;
  }
  if (msl_source_.empty()) {
    XELOGE("MslShader: SPIRV-Cross compilation produced empty output");
    return false;
  }

  // Get the entry point name that SPIRV-Cross chose.
  try {
    entry_point_name_ =
        compiler.get_cleansed_entry_point_name("main", execution_model);
  } catch (const std::exception& e) {
    XELOGE("MslShader: SPIRV-Cross entry point lookup failed: {}", e.what());
    entry_point_name_ = "main0";
  }
  if (entry_point_name_.empty()) {
    // Fallback — SPIRV-Cross often names it "main0".
    entry_point_name_ = "main0";
  }

  XELOGD("MslShader: Compiled SPIR-V to MSL ({} bytes, entry: {})",
         msl_source_.size(), entry_point_name_);

  // Compile MSL source to a Metal library.
  // Use a local autorelease pool to ensure NS::String factory objects and any
  // NS::Error autoreleased by the Metal driver are cleaned up promptly.
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  NS::Error* error = nullptr;
  auto* source_str =
      NS::String::string(msl_source_.c_str(), NS::UTF8StringEncoding);
  auto* compile_options = MTL::CompileOptions::alloc()->init();
  // Use fast math for better performance (matches Xbox 360 behavior better
  // than strict IEEE — the 360's ALU doesn't fully conform to IEEE anyway).
  compile_options->setFastMathEnabled(true);
  // Set the MSL language version to match what SPIRV-Cross generates (2.4).
  // Without this, the Metal compiler uses the OS default, which could reject
  // MSL 2.4 features on older OS versions or accept newer syntax on newer OS
  // versions — causing inconsistent behavior.
  compile_options->setLanguageVersion(MTL::LanguageVersion2_4);

  metal_library_ = device->newLibrary(source_str, compile_options, &error);
  compile_options->release();

  if (!metal_library_) {
    if (error) {
      XELOGE("MslShader: Metal library compilation failed: {}",
             error->localizedDescription()->utf8String());
    } else {
      XELOGE("MslShader: Metal library compilation failed (unknown error)");
    }
    pool->release();
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
    pool->release();
    return false;
  }

  pool->release();

  XELOGD("MslShader: Successfully created Metal function '{}'",
         entry_point_name_);
  return true;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
