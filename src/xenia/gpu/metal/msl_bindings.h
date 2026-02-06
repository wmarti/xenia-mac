/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_MSL_BINDINGS_H_
#define XENIA_GPU_METAL_MSL_BINDINGS_H_

#include <cstdint>

namespace xe {
namespace gpu {
namespace metal {

// Metal buffer/texture/sampler binding indices for the SPIRV-Cross MSL path.
// These must match the remappings applied in msl_shader.cc AddResourceBindings.
//
// The SPIR-V translator uses 4 descriptor sets:
//   Set 0: Shared memory and EDRAM (storage buffers)
//   Set 1: Constant buffers (system, float, bool/loop, fetch, clip, tess)
//   Set 2: Vertex stage textures
//   Set 3: Pixel stage textures
//
// SPIRV-Cross with argument_buffers=false maps each binding to a direct
// Metal buffer/texture/sampler index. We control these indices via
// add_msl_resource_binding().

namespace MslBufferIndex {
// Buffer indices for vertex and fragment shaders.
constexpr uint32_t kSharedMemory = 0;
constexpr uint32_t kSystemConstants = 1;
constexpr uint32_t kFloatConstantsVertex = 2;
constexpr uint32_t kFloatConstantsPixel = 3;
constexpr uint32_t kBoolLoopConstants = 4;
constexpr uint32_t kFetchConstants = 5;
constexpr uint32_t kClipPlaneConstants = 6;
constexpr uint32_t kTessellationConstants = 7;

// Total constant buffer slots used, for computing uniform buffer offsets.
constexpr uint32_t kConstantBufferCount = 8;

// Size of each constant buffer slice in the uniforms buffer (4KB, matching
// the D3D12/MSC layout for easy migration).
constexpr uint32_t kCbvSizeBytes = 4096;

// Total size of the uniforms buffer region for one stage (one draw).
// system(4KB) + float(4KB) + bool_loop(4KB) + fetch(4KB) = 16KB minimum.
// Allocate 5 * 4KB to match the existing MSC layout.
constexpr uint32_t kUniformsBytesPerStage = 5 * kCbvSizeBytes;
}  // namespace MslBufferIndex

namespace MslTextureIndex {
// Textures start at index 0 within each shader stage.
constexpr uint32_t kBase = 0;
// Maximum textures per stage (Xbox 360 has 32 fetch constants).
constexpr uint32_t kMaxPerStage = 32;
}  // namespace MslTextureIndex

namespace MslSamplerIndex {
// Samplers start at index 0 within each shader stage.
constexpr uint32_t kBase = 0;
// Maximum samplers per stage.
constexpr uint32_t kMaxPerStage = 16;
}  // namespace MslSamplerIndex

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_MSL_BINDINGS_H_
