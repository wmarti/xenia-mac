/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <metal_stdlib>

using namespace metal;

// Compute shader to downscale scaled resolve buffer data back to 1x resolution.
// Operates on 32x32 tiled data format used by Xbox 360.
// Each thread handles one output pixel (one 32x32 tile = 1024 threads).
//
// By default, picks the top-left pixel of each scale_x * scale_y block.
// When xe_downscale_half_pixel_offset is set, samples from (scale/2, scale/2)
// within each block instead.

struct XeResolveDownscaleConstants {
  uint xe_downscale_scale_x;         // 1 to kMaxDrawResolutionScaleAlongAxis
  uint xe_downscale_scale_y;         // 1 to kMaxDrawResolutionScaleAlongAxis
  uint xe_downscale_pixel_size_log2; // 0=8bit, 1=16bit, 2=32bit, 3=64bit
  uint xe_downscale_tile_count;      // Number of 32x32 tiles to process
  uint xe_downscale_source_offset_bytes; // Byte offset into source buffer
  // When non-zero, sample from (scale/2, scale/2) within each scaled block
  // instead of (0, 0).
  uint xe_downscale_half_pixel_offset;
};

static inline uint xe_load_u32(const device uint* buffer, uint byte_offset) {
  return buffer[byte_offset >> 2];
}

static inline void xe_store_u32(device uint* buffer, uint byte_offset,
                                uint value) {
  buffer[byte_offset >> 2] = value;
}

kernel void entry_xe(
    constant XeResolveDownscaleConstants& constants [[buffer(0)]],
    const device uint* xe_resolve_source [[buffer(1)]],
    device uint* xe_resolve_dest [[buffer(2)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]]) {
  uint tile_index = group_id.x;
  if (tile_index >= constants.xe_downscale_tile_count) {
    return;
  }

  uint row = thread_id.y;
  uint column = thread_id.x;
  uint pixel_index = row * 32u + column;  // 0-1023

  uint pixel_size = 1u << constants.xe_downscale_pixel_size_log2;
  uint tile_size_1x = 32u * 32u * pixel_size;
  uint scale_xy =
      constants.xe_downscale_scale_x * constants.xe_downscale_scale_y;
  uint tile_size_scaled = tile_size_1x * scale_xy;

  uint block_sample_offset = 0u;
  if (constants.xe_downscale_half_pixel_offset != 0u && scale_xy > 1u) {
    uint offset_x = constants.xe_downscale_scale_x >> 1u;
    uint offset_y = constants.xe_downscale_scale_y >> 1u;
    block_sample_offset =
        offset_x + offset_y * constants.xe_downscale_scale_x;
  }

  uint base_src_byte_offset =
      constants.xe_downscale_source_offset_bytes +
      tile_index * tile_size_scaled +
      block_sample_offset * pixel_size;

  switch (constants.xe_downscale_pixel_size_log2) {
    case 0u: {  // 8-bit - pack 4 bytes into one uint per 4 pixels.
      if ((pixel_index & 3u) != 0u) {
        return;
      }
      uint pack_index = pixel_index >> 2u;  // Which uint (0-255)
      uint dst_offset = tile_index * (32u * 32u) + pack_index * 4u;
      uint packed = 0u;
      for (uint i = 0u; i < 4u; ++i) {
        uint pixel_i = pixel_index + i;
        uint src_byte_offset =
            base_src_byte_offset + pixel_i * pixel_size * scale_xy;
        uint src_word = xe_load_u32(xe_resolve_source, src_byte_offset & ~3u);
        uint byte_val =
            (src_word >> ((src_byte_offset & 3u) * 8u)) & 0xFFu;
        packed |= (byte_val << (i * 8u));
      }
      xe_store_u32(xe_resolve_dest, dst_offset, packed);
      break;
    }
    case 1u: {  // 16-bit - pack 2 shorts into one uint per 2 pixels.
      if ((pixel_index & 1u) != 0u) {
        return;
      }
      uint pack_index = pixel_index >> 1u;  // Which uint (0-511)
      uint dst_offset = tile_index * (32u * 32u * 2u) + pack_index * 4u;
      uint packed = 0u;
      for (uint i = 0u; i < 2u; ++i) {
        uint pixel_i = pixel_index + i;
        uint src_byte_offset =
            base_src_byte_offset + pixel_i * pixel_size * scale_xy;
        uint src_word = xe_load_u32(xe_resolve_source, src_byte_offset & ~3u);
        uint short_val =
            (src_word >> ((src_byte_offset & 2u) * 8u)) & 0xFFFFu;
        packed |= (short_val << (i * 16u));
      }
      xe_store_u32(xe_resolve_dest, dst_offset, packed);
      break;
    }
    case 2u: {  // 32-bit - direct copy.
      uint src_byte_offset =
          base_src_byte_offset + pixel_index * pixel_size * scale_xy;
      uint dst_byte_offset =
          tile_index * tile_size_1x + pixel_index * pixel_size;
      xe_store_u32(xe_resolve_dest, dst_byte_offset,
                   xe_load_u32(xe_resolve_source, src_byte_offset));
      break;
    }
    case 3u: {  // 64-bit - direct copy (2 uints).
      uint src_byte_offset =
          base_src_byte_offset + pixel_index * pixel_size * scale_xy;
      uint dst_byte_offset =
          tile_index * tile_size_1x + pixel_index * pixel_size;
      xe_store_u32(xe_resolve_dest, dst_byte_offset,
                   xe_load_u32(xe_resolve_source, src_byte_offset));
      xe_store_u32(xe_resolve_dest, dst_byte_offset + 4u,
                   xe_load_u32(xe_resolve_source, src_byte_offset + 4u));
      break;
    }
  }
}
