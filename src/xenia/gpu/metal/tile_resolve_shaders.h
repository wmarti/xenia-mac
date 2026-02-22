/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_TILE_RESOLVE_SHADERS_H_
#define XENIA_GPU_METAL_TILE_RESOLVE_SHADERS_H_

#include <cstdint>

namespace xe {
namespace gpu {
namespace metal {

// MSL source for tile resolve shaders compiled at runtime.
// Tile shaders require MTLTileRenderPipelineDescriptor which needs
// attachment format info, so they cannot be pre-compiled to metallib.
//
// Constants structure must match XeTileResolveConstants in C++ exactly.
inline constexpr const char* kTileResolveShaderSource = R"msl(
#include <metal_stdlib>
using namespace metal;

struct XeTileResolveConstants {
  uint edram_info;
  uint coordinate_info;
  uint dest_info;
  uint dest_coordinate_info;
  uint dest_base;
  uint dest_endian;
  uint resolve_width;
  uint resolve_height;
  uint src_color_index;
  uint sample_select;
  uint do_clear;
  uint clear_value_lo;
  uint clear_value_hi;
  uint dest_format;
  uint dest_pitch_div_32;
  uint dest_offset_x_div_8;
  uint dest_offset_y_div_8;
  uint edram_base_tiles;
  uint edram_pitch_tiles;
  uint format_is_64bpp;
};

static inline uint xe_endian_swap_32(uint value, uint endian) {
  switch (endian) {
    case 1u:
      return ((value & 0x00FF00FFu) << 8u) | ((value & 0xFF00FF00u) >> 8u);
    case 2u:
      return ((value & 0x0000FFFFu) << 16u) | ((value >> 16u) & 0x0000FFFFu);
    case 3u:
      return ((value & 0x000000FFu) << 24u) |
             ((value & 0x0000FF00u) << 8u) |
             ((value & 0x00FF0000u) >> 8u) |
             ((value >> 24u) & 0x000000FFu);
    default:
      return value;
  }
}

// Pack a float4 color into a 32-bit value for the guest memory destination.
// Uses xenos::ColorFormat values (from RB_COPY_DEST_INFO.copy_dest_format):
//   k_8_8_8_8 = 6, k_2_10_10_10 = 7, k_8_8 = 10,
//   k_16_16 = 25, k_16_16_FLOAT = 31, k_32_FLOAT = 36, etc.
static inline uint xe_pack_color_32bpp(float4 color, uint format) {
  switch (format) {
    case 6u:    // k_8_8_8_8
    case 14u: { // k_8_8_8_8_A
      uint4 c = uint4(saturate(color) * 255.0f + 0.5f);
      return c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
    }
    case 7u: { // k_2_10_10_10
      uint r = uint(saturate(color.r) * 1023.0f + 0.5f);
      uint g = uint(saturate(color.g) * 1023.0f + 0.5f);
      uint b = uint(saturate(color.b) * 1023.0f + 0.5f);
      uint a = uint(saturate(color.a) * 3.0f + 0.5f);
      return r | (g << 10u) | (b << 20u) | (a << 30u);
    }
    case 10u: { // k_8_8
      uint r = uint(saturate(color.r) * 255.0f + 0.5f);
      uint g = uint(saturate(color.g) * 255.0f + 0.5f);
      return r | (g << 8u);
    }
    case 25u: { // k_16_16 (signed normalized)
      int2 c = int2(clamp(color.rg, float2(-1.0f), float2(1.0f)) * 32767.0f +
                     select(float2(-0.5f), float2(0.5f),
                            color.rg >= float2(0.0f)));
      return (uint(c.r) & 0xFFFFu) | (uint(c.g) << 16u);
    }
    case 31u: { // k_16_16_FLOAT
      half2 h = half2(color.rg);
      return uint(as_type<ushort>(h.x)) | (uint(as_type<ushort>(h.y)) << 16u);
    }
    case 36u: { // k_32_FLOAT
      return as_type<uint>(color.r);
    }
    case 3u: { // k_1_5_5_5
      uint r = uint(saturate(color.r) * 31.0f + 0.5f);
      uint g = uint(saturate(color.g) * 31.0f + 0.5f);
      uint b = uint(saturate(color.b) * 31.0f + 0.5f);
      uint a = color.a >= 0.5f ? 1u : 0u;
      return a | (r << 1u) | (g << 6u) | (b << 11u);
    }
    case 4u: { // k_5_6_5
      uint r = uint(saturate(color.r) * 31.0f + 0.5f);
      uint g = uint(saturate(color.g) * 63.0f + 0.5f);
      uint b = uint(saturate(color.b) * 31.0f + 0.5f);
      return r | (g << 5u) | (b << 11u);
    }
    case 15u: { // k_4_4_4_4
      uint r = uint(saturate(color.r) * 15.0f + 0.5f);
      uint g = uint(saturate(color.g) * 15.0f + 0.5f);
      uint b = uint(saturate(color.b) * 15.0f + 0.5f);
      uint a = uint(saturate(color.a) * 15.0f + 0.5f);
      return r | (g << 4u) | (b << 8u) | (a << 12u);
    }
    default: {
      // Fallback: pack as 8_8_8_8.
      uint4 c = uint4(saturate(color) * 255.0f + 0.5f);
      return c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
    }
  }
}

static inline uint xe_resolve_dest_address(uint2 pixel, uint pitch_div_32,
                                           uint offset_x_div_8,
                                           uint offset_y_div_8,
                                           uint dest_base) {
  uint2 dest_pixel = pixel + uint2(offset_x_div_8 * 8u, offset_y_div_8 * 8u);
  uint pitch = pitch_div_32 * 32u;
  uint bytes_per_pixel = 4u;
  return dest_base + (dest_pixel.y * pitch + dest_pixel.x) * bytes_per_pixel;
}

struct TileResolveIB0 { float4 color [[color(0)]]; };
struct TileResolveIB1 { float4 color [[color(1)]]; };
struct TileResolveIB2 { float4 color [[color(2)]]; };
struct TileResolveIB3 { float4 color [[color(3)]]; };

kernel void xe_tile_resolve_color0(
    imageblock<TileResolveIB0, imageblock_layout_implicit> img,
    constant XeTileResolveConstants& c [[buffer(0)]],
    device uint* dest [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgid [[threadgroup_position_in_grid]]) {
  uint2 pixel = uint2(tgid) * uint2(32u, 32u) + uint2(tid);
  if (pixel.x >= c.resolve_width || pixel.y >= c.resolve_height) return;
  auto data = img.read(tid);
  float4 color = data.color;
  uint packed = xe_pack_color_32bpp(color, c.dest_format);
  packed = xe_endian_swap_32(packed, c.dest_endian);
  uint addr = xe_resolve_dest_address(pixel, c.dest_pitch_div_32,
      c.dest_offset_x_div_8, c.dest_offset_y_div_8, c.dest_base);
  dest[addr >> 2u] = packed;
}

kernel void xe_tile_resolve_color1(
    imageblock<TileResolveIB1, imageblock_layout_implicit> img,
    constant XeTileResolveConstants& c [[buffer(0)]],
    device uint* dest [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgid [[threadgroup_position_in_grid]]) {
  uint2 pixel = uint2(tgid) * uint2(32u, 32u) + uint2(tid);
  if (pixel.x >= c.resolve_width || pixel.y >= c.resolve_height) return;
  auto data = img.read(tid);
  float4 color = data.color;
  uint packed = xe_pack_color_32bpp(color, c.dest_format);
  packed = xe_endian_swap_32(packed, c.dest_endian);
  uint addr = xe_resolve_dest_address(pixel, c.dest_pitch_div_32,
      c.dest_offset_x_div_8, c.dest_offset_y_div_8, c.dest_base);
  dest[addr >> 2u] = packed;
}

kernel void xe_tile_resolve_color2(
    imageblock<TileResolveIB2, imageblock_layout_implicit> img,
    constant XeTileResolveConstants& c [[buffer(0)]],
    device uint* dest [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgid [[threadgroup_position_in_grid]]) {
  uint2 pixel = uint2(tgid) * uint2(32u, 32u) + uint2(tid);
  if (pixel.x >= c.resolve_width || pixel.y >= c.resolve_height) return;
  auto data = img.read(tid);
  float4 color = data.color;
  uint packed = xe_pack_color_32bpp(color, c.dest_format);
  packed = xe_endian_swap_32(packed, c.dest_endian);
  uint addr = xe_resolve_dest_address(pixel, c.dest_pitch_div_32,
      c.dest_offset_x_div_8, c.dest_offset_y_div_8, c.dest_base);
  dest[addr >> 2u] = packed;
}

kernel void xe_tile_resolve_color3(
    imageblock<TileResolveIB3, imageblock_layout_implicit> img,
    constant XeTileResolveConstants& c [[buffer(0)]],
    device uint* dest [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgid [[threadgroup_position_in_grid]]) {
  uint2 pixel = uint2(tgid) * uint2(32u, 32u) + uint2(tid);
  if (pixel.x >= c.resolve_width || pixel.y >= c.resolve_height) return;
  auto data = img.read(tid);
  float4 color = data.color;
  uint packed = xe_pack_color_32bpp(color, c.dest_format);
  packed = xe_endian_swap_32(packed, c.dest_endian);
  uint addr = xe_resolve_dest_address(pixel, c.dest_pitch_div_32,
      c.dest_offset_x_div_8, c.dest_offset_y_div_8, c.dest_base);
  dest[addr >> 2u] = packed;
}
)msl";

// C++ mirror of the tile resolve constants structure.
struct XeTileResolveConstants {
  uint32_t edram_info;
  uint32_t coordinate_info;
  uint32_t dest_info;
  uint32_t dest_coordinate_info;
  uint32_t dest_base;
  uint32_t dest_endian;
  uint32_t resolve_width;
  uint32_t resolve_height;
  uint32_t src_color_index;
  uint32_t sample_select;
  uint32_t do_clear;
  uint32_t clear_value_lo;
  uint32_t clear_value_hi;
  uint32_t dest_format;
  uint32_t dest_pitch_div_32;
  uint32_t dest_offset_x_div_8;
  uint32_t dest_offset_y_div_8;
  uint32_t edram_base_tiles;
  uint32_t edram_pitch_tiles;
  uint32_t format_is_64bpp;
};

// Function names for each color attachment index.
inline constexpr const char* kTileResolveFunctionNames[4] = {
    "xe_tile_resolve_color0",
    "xe_tile_resolve_color1",
    "xe_tile_resolve_color2",
    "xe_tile_resolve_color3",
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_TILE_RESOLVE_SHADERS_H_
