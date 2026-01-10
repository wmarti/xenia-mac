/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GPU_FLAGS_H_
#define XENIA_GPU_GPU_FLAGS_H_
#include "xenia/base/cvar.h"

DECLARE_path(trace_gpu_prefix);
DECLARE_bool(trace_gpu_stream);

DECLARE_path(dump_shaders);

DECLARE_bool(vsync);

DECLARE_bool(gpu_allow_invalid_fetch_constants);

DECLARE_bool(non_seamless_cube_map);
DECLARE_bool(half_pixel_offset);

// Metal-specific debug flag: make EDRAM buffer CPU-visible so its contents
// can be inspected from resolve code. Only use in debug builds.
DECLARE_bool(metal_debug_edram_cpu_visible);
// Metal: enable ROV-style EDRAM blending path (ordered blending).
DECLARE_bool(metal_edram_rov);
DECLARE_bool(metal_edram_compute_fallback);
DECLARE_bool(metal_disable_resolve_edram_dump);
DECLARE_bool(metal_disable_transfer_shaders);
DECLARE_bool(metal_edram_blend_bounds_check);
DECLARE_bool(metal_disable_resolve_pitch_override);
DECLARE_bool(metal_log_resolve_copy_dest_info);
DECLARE_bool(metal_log_copy_dest_register_writes);
DECLARE_bool(metal_force_full_scissor_on_swap_resolve);
DECLARE_bool(metal_log_edram_dump_color_samples);

// Enable extremely verbose Metal backend logging (may significantly impact performance).
DECLARE_bool(metal_verbose_logging);

DECLARE_bool(metal_shader_disk_cache);
DECLARE_bool(metal_pipeline_binary_archive);
DECLARE_bool(metal_pipeline_disk_cache);
DECLARE_int32(metal_draw_ring_count);
DECLARE_bool(metal_use_heaps);
DECLARE_bool(metal_shared_memory_zero_copy);
DECLARE_int32(metal_heap_min_bytes);
DECLARE_bool(metal_log_cache_stats);
DECLARE_int32(metal_log_cache_stats_interval_seconds);
DECLARE_bool(metal_texture_cache_use_private);
DECLARE_bool(metal_texture_upload_via_blit);

DECLARE_int32(query_occlusion_fake_sample_count);

#endif  // XENIA_GPU_GPU_FLAGS_H_
