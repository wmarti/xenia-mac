/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Tile resolve shader for Apple TBDR GPUs.
//
// Runs as a tile dispatch within the current render pass, reading directly from
// tile memory via implicit imageblocks (render pass color attachments) and
// writing the resolved color data to guest shared memory.
//
// This avoids breaking the render pass (and the associated tile memory
// store/reload cost) for color resolves on Apple Silicon.

#include <metal_stdlib>
using namespace metal;

// Must match draw_util::ResolveCopyShaderConstants layout exactly.
// We pass only the DestRelative portion (no dest_base) plus dest_base
// separately for the non-scaled path.
struct XeTileResolveConstants {
  // ResolveEdramInfo (packed uint32_t).
  uint edram_info;
  // ResolveCoordinateInfo (packed uint32_t).
  uint coordinate_info;
  // RB_COPY_DEST_INFO (packed uint32_t).
  uint dest_info;
  // ResolveCopyDestCoordinateInfo (packed uint32_t).
  uint dest_coordinate_info;
  // Destination base address in shared memory (bytes).
  uint dest_base;
  // Destination endianness (from RB_COPY_DEST_INFO).
  uint dest_endian;
  // Width of resolve rect in pixels.
  uint resolve_width;
  // Height of resolve rect in pixels.
  uint resolve_height;
  // Source render target attachment index (0-3).
  uint src_color_index;
  // Sample select mode: 0=averaged resolve, 1..4=specific sample.
  uint sample_select;
  // 1 if the resolve clear should be performed after copy.
  uint do_clear;
  // Clear color (two uint32s for 64bpp support).
  uint clear_value_lo;
  uint clear_value_hi;
  // Destination format (xenos::ColorFormat packed).
  uint dest_format;
  // Pitch of destination in 32-pixel units.
  uint dest_pitch_div_32;
  // Offset X/Y of destination in 8-pixel units.
  uint dest_offset_x_div_8;
  uint dest_offset_y_div_8;
  // EDRAM base tiles.
  uint edram_base_tiles;
  // EDRAM pitch tiles.
  uint edram_pitch_tiles;
  // Whether the source is 64bpp.
  uint format_is_64bpp;
};

// Endian swap a 32-bit value based on Xenos endianness mode.
static inline uint xe_endian_swap_32(uint value, uint endian) {
  // Endian modes: 0=none, 1=swap bytes in 16-bit words, 2=swap 16-bit words,
  // 3=swap both (full byte reversal).
  switch (endian) {
    case 1u:
      // 8-in-16: swap bytes within each 16-bit half.
      return ((value & 0x00FF00FFu) << 8u) | ((value & 0xFF00FF00u) >> 8u);
    case 2u:
      // 8-in-32: swap 16-bit halves.
      return ((value & 0x0000FFFFu) << 16u) | ((value >> 16u) & 0x0000FFFFu);
    case 3u:
      // Full byte reversal.
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

// Compute the destination byte address in guest tiled texture memory.
// Xbox 360 textures use a tiled memory layout (32x32 pixel tiles for 32bpp).
static inline uint xe_resolve_dest_address(uint2 pixel, uint pitch_div_32,
                                           uint offset_x_div_8,
                                           uint offset_y_div_8,
                                           uint dest_base,
                                           uint bpp_log2) {
  // Destination pixel coordinates including offset.
  uint2 dest_pixel = pixel + uint2(offset_x_div_8 * 8u, offset_y_div_8 * 8u);
  uint pitch = pitch_div_32 * 32u;

  // Simple linear addressing for the resolve destination.
  // The actual tiling is handled by the shared memory system.
  uint bytes_per_pixel = 1u << bpp_log2;
  return dest_base + (dest_pixel.y * pitch + dest_pixel.x) * bytes_per_pixel;
}

// ===== Tile kernel entry points =====
// One kernel per source color attachment index (0-3).
// Using implicit imageblocks to read from render pass color attachments.

// Imageblock structure for each color attachment.
struct TileResolveIB0 { float4 color [[color(0)]]; };
struct TileResolveIB1 { float4 color [[color(1)]]; };
struct TileResolveIB2 { float4 color [[color(2)]]; };
struct TileResolveIB3 { float4 color [[color(3)]]; };

// Tile kernel for resolving color attachment 0.
kernel void xe_tile_resolve_color0(
    imageblock<TileResolveIB0, imageblock_layout_implicit> img,
    constant XeTileResolveConstants& c [[buffer(0)]],
    device uint* dest [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgid [[threadgroup_position_in_grid]]) {
  // Compute absolute pixel coordinate.
  uint2 pixel = uint2(tgid) * uint2(32u, 32u) + uint2(tid);

  // Early-out if outside the resolve rectangle.
  if (pixel.x >= c.resolve_width || pixel.y >= c.resolve_height) {
    return;
  }

  // Read from tile memory (implicit imageblock).
  auto data = img.read(tid);
  float4 color = data.color;

  // Pack and write to destination.
  uint packed = xe_pack_color_32bpp(color, c.dest_format);
  packed = xe_endian_swap_32(packed, c.dest_endian);

  uint addr = xe_resolve_dest_address(
      pixel, c.dest_pitch_div_32, c.dest_offset_x_div_8,
      c.dest_offset_y_div_8, c.dest_base, 2u);
  dest[addr >> 2u] = packed;
}

// Tile kernel for resolving color attachment 1.
kernel void xe_tile_resolve_color1(
    imageblock<TileResolveIB1, imageblock_layout_implicit> img,
    constant XeTileResolveConstants& c [[buffer(0)]],
    device uint* dest [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgid [[threadgroup_position_in_grid]]) {
  uint2 pixel = uint2(tgid) * uint2(32u, 32u) + uint2(tid);
  if (pixel.x >= c.resolve_width || pixel.y >= c.resolve_height) {
    return;
  }
  auto data = img.read(tid);
  float4 color = data.color;
  uint packed = xe_pack_color_32bpp(color, c.dest_format);
  packed = xe_endian_swap_32(packed, c.dest_endian);
  uint addr = xe_resolve_dest_address(
      pixel, c.dest_pitch_div_32, c.dest_offset_x_div_8,
      c.dest_offset_y_div_8, c.dest_base, 2u);
  dest[addr >> 2u] = packed;
}

// Tile kernel for resolving color attachment 2.
kernel void xe_tile_resolve_color2(
    imageblock<TileResolveIB2, imageblock_layout_implicit> img,
    constant XeTileResolveConstants& c [[buffer(0)]],
    device uint* dest [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgid [[threadgroup_position_in_grid]]) {
  uint2 pixel = uint2(tgid) * uint2(32u, 32u) + uint2(tid);
  if (pixel.x >= c.resolve_width || pixel.y >= c.resolve_height) {
    return;
  }
  auto data = img.read(tid);
  float4 color = data.color;
  uint packed = xe_pack_color_32bpp(color, c.dest_format);
  packed = xe_endian_swap_32(packed, c.dest_endian);
  uint addr = xe_resolve_dest_address(
      pixel, c.dest_pitch_div_32, c.dest_offset_x_div_8,
      c.dest_offset_y_div_8, c.dest_base, 2u);
  dest[addr >> 2u] = packed;
}

// Tile kernel for resolving color attachment 3.
kernel void xe_tile_resolve_color3(
    imageblock<TileResolveIB3, imageblock_layout_implicit> img,
    constant XeTileResolveConstants& c [[buffer(0)]],
    device uint* dest [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgid [[threadgroup_position_in_grid]]) {
  uint2 pixel = uint2(tgid) * uint2(32u, 32u) + uint2(tid);
  if (pixel.x >= c.resolve_width || pixel.y >= c.resolve_height) {
    return;
  }
  auto data = img.read(tid);
  float4 color = data.color;
  uint packed = xe_pack_color_32bpp(color, c.dest_format);
  packed = xe_endian_swap_32(packed, c.dest_endian);
  uint addr = xe_resolve_dest_address(
      pixel, c.dest_pitch_div_32, c.dest_offset_x_div_8,
      c.dest_offset_y_div_8, c.dest_base, 2u);
  dest[addr >> 2u] = packed;
}
