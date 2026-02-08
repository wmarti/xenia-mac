/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_MSL_TESS_FACTOR_KERNELS_H_
#define XENIA_GPU_METAL_MSL_TESS_FACTOR_KERNELS_H_

// Embedded MSL compute kernels for tessellation factor generation.
// These are compiled at runtime by MetalCommandProcessor::InitializeMsl-
// Tessellation() and dispatched before drawPatches() for tessellated draws.
//
// NOTE: This header is #include'd inside namespace xe::gpu::metal in
// metal_command_processor.cc — do NOT add namespace declarations here.

// --------------------------------------------------------------------
// Uniform tessellation factor kernels (discrete / continuous modes).
// All patches receive the same factor from xe_tessellation_factor_range.y.
// --------------------------------------------------------------------

static const char* kMslTessFactorUniformTriangle = R"msl(
#include <metal_stdlib>
using namespace metal;

struct TessFactorParams {
  float edge_factor;
  float inside_factor;
  uint  patch_count;
};

kernel void tess_factor_triangle(
    device MTLTriangleTessellationFactorsHalf* factors [[buffer(0)]],
    constant TessFactorParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
  if (tid >= params.patch_count) return;
  half ef = half(params.edge_factor);
  half inf = half(params.inside_factor);
  factors[tid].edgeTessellationFactor[0] = ef;
  factors[tid].edgeTessellationFactor[1] = ef;
  factors[tid].edgeTessellationFactor[2] = ef;
  factors[tid].insideTessellationFactor = inf;
}
)msl";

static const char* kMslTessFactorUniformQuad = R"msl(
#include <metal_stdlib>
using namespace metal;

struct TessFactorParams {
  float edge_factor;
  float inside_factor;
  uint  patch_count;
};

kernel void tess_factor_quad(
    device MTLQuadTessellationFactorsHalf* factors [[buffer(0)]],
    constant TessFactorParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
  if (tid >= params.patch_count) return;
  half ef = half(params.edge_factor);
  half inf = half(params.inside_factor);
  factors[tid].edgeTessellationFactor[0] = ef;
  factors[tid].edgeTessellationFactor[1] = ef;
  factors[tid].edgeTessellationFactor[2] = ef;
  factors[tid].edgeTessellationFactor[3] = ef;
  factors[tid].insideTessellationFactor[0] = inf;
  factors[tid].insideTessellationFactor[1] = inf;
}
)msl";

// --------------------------------------------------------------------
// Adaptive tessellation factor kernels.
// Per-patch edge factors are read from shared memory (the guest "index
// buffer" repurposed as a factor buffer for adaptive tessellation).
//
// Each factor is a big-endian float32.  The kernel endian-swaps, adds
// 1.0 (Xbox 360 convention), clamps to [factor_min, factor_max], maps
// the Xbox 360 edge order to Metal's edge order, and computes inside
// factors as the minimum of opposing (quad) or all (triangle) edges.
//
// References:
//   - adaptive_triangle.hs.glsl / adaptive_quad.hs.glsl (Vulkan path)
//   - https://www.slideshare.net/blackdevilvikas/
//         next-generation-graphics-programming-on-xbox-360
//   - http://www.uraldev.ru/files/download/21/
//         Real-Time_Tessellation_on_GPU.pdf  (Code Listing 15)
// --------------------------------------------------------------------

static const char* kMslTessFactorAdaptiveTriangle = R"msl(
#include <metal_stdlib>
using namespace metal;

struct AdaptiveTessParams {
  uint  factor_base_dwords;   // Offset into shared_memory (in uint32 words).
  uint  patch_count;
  float factor_min;           // Clamping range (1.0 already added on CPU).
  float factor_max;
  uint  vertex_index_endian;  // 0 = none, 1 = 8in16, 2 = 8in32, 3 = 16in32.
};

// Endian-swap a 32-bit word per the Xenos endian mode.
uint endian_swap(uint v, uint mode) {
  switch (mode) {
    case 1u:  // 8-in-16
      return ((v & 0x00FF00FFu) << 8u) | ((v & 0xFF00FF00u) >> 8u);
    case 2u:  // 8-in-32
      return ((v & 0x000000FFu) << 24u) | ((v & 0x0000FF00u) << 8u) |
             ((v & 0x00FF0000u) >> 8u)  | ((v & 0xFF000000u) >> 24u);
    case 3u:  // 16-in-32
      return ((v & 0x0000FFFFu) << 16u) | ((v & 0xFFFF0000u) >> 16u);
    default:  // 0 = none
      return v;
  }
}

kernel void tess_factor_adaptive_triangle(
    device const uint* shared_memory [[buffer(0)]],
    device MTLTriangleTessellationFactorsHalf* factors [[buffer(1)]],
    constant AdaptiveTessParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
  if (tid >= params.patch_count) return;

  // Read 3 edge factors (big-endian float32) from shared memory.
  float edge_factors[3];
  for (uint i = 0; i < 3; i++) {
    uint raw = shared_memory[params.factor_base_dwords + tid * 3u + i];
    raw = endian_swap(raw, params.vertex_index_endian);
    // Add 1.0 per Xbox 360 convention, then clamp.
    float f = as_type<float>(raw) + 1.0f;
    edge_factors[i] = clamp(f, params.factor_min, params.factor_max);
  }

  // Map Xbox 360 edge order to Metal (same as D3D12/Vulkan):
  //   Xbox 360: [0] = v0->v1,  [1] = v1->v2,  [2] = v2->v0
  //   Metal:    edge[0] = U0 (v1->v2), edge[1] = V0 (v2->v0),
  //             edge[2] = W0 (v0->v1)
  factors[tid].edgeTessellationFactor[0] = half(edge_factors[1]);  // v1->v2
  factors[tid].edgeTessellationFactor[1] = half(edge_factors[2]);  // v2->v0
  factors[tid].edgeTessellationFactor[2] = half(edge_factors[0]);  // v0->v1

  // Inside factor = minimum of all edge factors.
  factors[tid].insideTessellationFactor = half(min(min(
      edge_factors[0], edge_factors[1]), edge_factors[2]));
}
)msl";

static const char* kMslTessFactorAdaptiveQuad = R"msl(
#include <metal_stdlib>
using namespace metal;

struct AdaptiveTessParams {
  uint  factor_base_dwords;
  uint  patch_count;
  float factor_min;
  float factor_max;
  uint  vertex_index_endian;
};

uint endian_swap(uint v, uint mode) {
  switch (mode) {
    case 1u:
      return ((v & 0x00FF00FFu) << 8u) | ((v & 0xFF00FF00u) >> 8u);
    case 2u:
      return ((v & 0x000000FFu) << 24u) | ((v & 0x0000FF00u) << 8u) |
             ((v & 0x00FF0000u) >> 8u)  | ((v & 0xFF000000u) >> 24u);
    case 3u:
      return ((v & 0x0000FFFFu) << 16u) | ((v & 0xFFFF0000u) >> 16u);
    default:
      return v;
  }
}

kernel void tess_factor_adaptive_quad(
    device const uint* shared_memory [[buffer(0)]],
    device MTLQuadTessellationFactorsHalf* factors [[buffer(1)]],
    constant AdaptiveTessParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
  if (tid >= params.patch_count) return;

  // Read 4 edge factors (big-endian float32) from shared memory.
  float edge_factors[4];
  for (uint i = 0; i < 4; i++) {
    uint raw = shared_memory[params.factor_base_dwords + tid * 4u + i];
    raw = endian_swap(raw, params.vertex_index_endian);
    float f = as_type<float>(raw) + 1.0f;
    edge_factors[i] = clamp(f, params.factor_min, params.factor_max);
  }

  // Map Xbox 360 edge order to Metal (same as D3D12/Vulkan):
  //   Xbox 360 factors go around the perimeter:
  //     [0] = U0V0->U1V0, [1] = U1V0->U1V1,
  //     [2] = U1V1->U0V1, [3] = U0V1->U0V0
  //   Metal/Vulkan: edge[i] = input[(i+3) & 3]
  //     edge[0] = between U0V1 and U0V0
  //     edge[1] = between U0V0 and U1V0
  //     edge[2] = between U1V0 and U1V1
  //     edge[3] = between U1V1 and U0V1
  factors[tid].edgeTessellationFactor[0] = half(edge_factors[3]);
  factors[tid].edgeTessellationFactor[1] = half(edge_factors[0]);
  factors[tid].edgeTessellationFactor[2] = half(edge_factors[1]);
  factors[tid].edgeTessellationFactor[3] = half(edge_factors[2]);

  // Inside factors: minimum of opposing edge pairs.
  // Vulkan/Metal: inside[0] along U, inside[1] along V.
  float mapped_edge_1 = edge_factors[0];  // Metal edge[1]
  float mapped_edge_3 = edge_factors[2];  // Metal edge[3]
  float mapped_edge_0 = edge_factors[3];  // Metal edge[0]
  float mapped_edge_2 = edge_factors[1];  // Metal edge[2]
  factors[tid].insideTessellationFactor[0] =
      half(min(mapped_edge_1, mapped_edge_3));
  factors[tid].insideTessellationFactor[1] =
      half(min(mapped_edge_0, mapped_edge_2));
}
)msl";

#endif  // XENIA_GPU_METAL_MSL_TESS_FACTOR_KERNELS_H_
