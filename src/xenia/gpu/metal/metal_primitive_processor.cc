/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_primitive_processor.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "xenia/base/assert.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/metal/metal_command_processor.h"

namespace xe {
namespace gpu {
namespace metal {

MetalPrimitiveProcessor::MetalPrimitiveProcessor(
    MetalCommandProcessor& command_processor, const RegisterFile& register_file,
    Memory& memory, TraceWriter& trace_writer, SharedMemory& shared_memory)
    : PrimitiveProcessor(register_file, memory, trace_writer, shared_memory),
      command_processor_(command_processor) {}

MetalPrimitiveProcessor::~MetalPrimitiveProcessor() { Shutdown(true); }

bool MetalPrimitiveProcessor::Initialize() {
  // When using SPIRV-Cross (no mesh shaders / MSC), point sprites and
  // rectangle lists must be expanded in the vertex shader because there
  // are no geometry shaders available.  The SpirvShaderTranslator has
  // built-in support for kPointListAsTriangleStrip and
  // kRectangleListAsTriangleStrip host vertex shader types.
  bool spirvcross = cvars::metal_use_spirvcross;
  bool point_sprites_without_expansion = !spirvcross;
  bool rect_lists_without_expansion = !spirvcross;

  if (!InitializeCommon(true,   // full_32bit_vertex_indices_supported
                        false,  // triangle_fans_supported (will convert)
                        false,  // line_loops_supported (will convert)
                        false,  // quad_lists_supported (will convert)
                        point_sprites_without_expansion,
                        rect_lists_without_expansion)) {
    Shutdown();
    return false;
  }

  XELOGI(
      "MetalPrimitiveProcessor initialized (spirvcross={}, "
      "vs_point_expansion={}, vs_rect_expansion={})",
      spirvcross, !point_sprites_without_expansion,
      !rect_lists_without_expansion);

  if (spirvcross && !point_sprites_without_expansion &&
      !rect_lists_without_expansion) {
    // The generic primitive processor emits restart-separated triangle strips
    // for VS expansion. Keep a no-restart triangle-list fallback for Metal
    // SPIRV-Cross draws in case strip restart semantics diverge.
    constexpr uint32_t kMaxExpandedPrimitiveCount = UINT16_MAX;
    constexpr uint32_t kIndicesPerExpandedPrimitive = 6;
    size_t index_count =
        size_t(kMaxExpandedPrimitiveCount) * kIndicesPerExpandedPrimitive;
    size_t buffer_size_bytes = index_count * sizeof(uint32_t);
    MTL::Device* device = command_processor_.GetMetalDevice();
    expansion_triangle_list_index_buffer_ =
        device->newBuffer(buffer_size_bytes, MTL::ResourceStorageModeShared);
    if (!expansion_triangle_list_index_buffer_) {
      XELOGE(
          "Failed to create Metal expansion triangle-list fallback index "
          "buffer");
      Shutdown();
      return false;
    }
    expansion_triangle_list_index_buffer_->setLabel(NS::String::string(
        "Xenia Expansion Triangle List Index Buffer", NS::UTF8StringEncoding));
    uint32_t* indices = reinterpret_cast<uint32_t*>(
        expansion_triangle_list_index_buffer_->contents());
    for (uint32_t i = 0; i < kMaxExpandedPrimitiveCount; ++i) {
      uint32_t base = i << 2;
      size_t write_index = size_t(i) * kIndicesPerExpandedPrimitive;
      indices[write_index + 0] = base + 0;
      indices[write_index + 1] = base + 1;
      indices[write_index + 2] = base + 2;
      indices[write_index + 3] = base + 2;
      indices[write_index + 4] = base + 1;
      indices[write_index + 5] = base + 3;
    }
  }

  return true;
}

void MetalPrimitiveProcessor::Shutdown(bool from_destructor) {
  // Release all frame index buffers
  for (auto& frame_buffer : frame_index_buffers_) {
    if (frame_buffer.buffer) {
      frame_buffer.buffer->release();
    }
  }
  frame_index_buffers_.clear();

  // Release built-in index buffer
  if (builtin_index_buffer_) {
    builtin_index_buffer_->release();
    builtin_index_buffer_ = nullptr;
    builtin_index_buffer_gpu_address_ = 0;
    builtin_index_buffer_size_ = 0;
  }
  if (expansion_triangle_list_index_buffer_) {
    expansion_triangle_list_index_buffer_->release();
    expansion_triangle_list_index_buffer_ = nullptr;
  }

  if (!from_destructor) {
    ShutdownCommon();
  }
}

void MetalPrimitiveProcessor::CompletedSubmissionUpdated() {
  // Nothing to do for Metal
}

void MetalPrimitiveProcessor::BeginSubmission() {
  // Nothing to do for Metal
}

void MetalPrimitiveProcessor::BeginFrame() {
  converted_index_buffers_.clear();

  // Clean up old frame index buffers
  ++current_frame_;
  uint64_t current_frame = current_frame_;

  frame_index_buffers_.erase(
      std::remove_if(frame_index_buffers_.begin(), frame_index_buffers_.end(),
                     [current_frame](const FrameIndexBuffer& buffer) {
                       // Keep buffers used in the last 2 frames
                       if (current_frame - buffer.last_frame_used > 2) {
                         if (buffer.buffer) {
                           buffer.buffer->release();
                         }
                         return true;
                       }
                       return false;
                     }),
      frame_index_buffers_.end());
}

void MetalPrimitiveProcessor::EndFrame() {
  ClearPerFrameCache();
  converted_index_buffers_.clear();
}

MTL::Buffer* MetalPrimitiveProcessor::GetConvertedIndexBuffer(
    size_t handle, uint64_t& offset_bytes_out) const {
  if (handle >= converted_index_buffers_.size()) {
    XELOGE("Converted index buffer handle {} is out of range {}", handle,
           converted_index_buffers_.size());
    offset_bytes_out = 0;
    return nullptr;
  }

  const ConvertedIndexBufferBinding& binding = converted_index_buffers_[handle];
  offset_bytes_out = binding.offset_bytes;
  return binding.buffer;
}

bool MetalPrimitiveProcessor::InitializeBuiltinIndexBuffer(
    size_t size_bytes, std::function<void(void*)> fill_callback) {
  assert_not_zero(size_bytes);
  assert_null(builtin_index_buffer_);

  MTL::Device* device = command_processor_.GetMetalDevice();

  // Create buffer with shared storage so we can write to it
  builtin_index_buffer_ =
      device->newBuffer(size_bytes, MTL::ResourceStorageModeShared);
  if (!builtin_index_buffer_) {
    XELOGE("Failed to create Metal built-in index buffer");
    return false;
  }
  builtin_index_buffer_size_ = size_bytes;

  builtin_index_buffer_->setLabel(NS::String::string(
      "Xenia Built-in Index Buffer", NS::UTF8StringEncoding));

  // Fill the buffer with built-in indices
  void* buffer_data = builtin_index_buffer_->contents();
  fill_callback(buffer_data);

  // Get GPU address for binding
  builtin_index_buffer_gpu_address_ = builtin_index_buffer_->gpuAddress();

  XELOGI("Created Metal built-in index buffer ({} bytes)", size_bytes);
  return true;
}

void* MetalPrimitiveProcessor::RequestHostConvertedIndexBufferForCurrentFrame(
    xenos::IndexFormat format, uint32_t index_count, bool coalign_for_simd,
    uint32_t coalignment_original_address, size_t& backend_handle_out) {
  // Calculate required size
  size_t element_size = format == xenos::IndexFormat::kInt16 ? sizeof(uint16_t)
                                                             : sizeof(uint32_t);
  size_t required_size = index_count * element_size;

  // Add padding for SIMD alignment if requested
  if (coalign_for_simd) {
    required_size += XE_GPU_PRIMITIVE_PROCESSOR_SIMD_SIZE;
  }

  // Find or create a buffer large enough
  FrameIndexBuffer* chosen_buffer = nullptr;
  uint64_t current_frame = current_frame_;

  // First try to find an existing buffer that's large enough
  for (auto& frame_buffer : frame_index_buffers_) {
    if (frame_buffer.size >= required_size &&
        frame_buffer.last_frame_used != current_frame) {
      chosen_buffer = &frame_buffer;
      break;
    }
  }

  // If no suitable buffer found, create a new one
  if (!chosen_buffer) {
    MTL::Device* device = command_processor_.GetMetalDevice();

    // Round up to next power of 2 for better reuse
    size_t allocation_size = required_size;
    allocation_size = std::max(allocation_size, size_t(4096));
    allocation_size = (allocation_size + 4095) & ~4095;  // Round to 4KB

    MTL::Buffer* new_buffer =
        device->newBuffer(allocation_size, MTL::ResourceStorageModeShared);

    if (!new_buffer) {
      XELOGE("Failed to create Metal index buffer for primitive conversion");
      backend_handle_out = 0;
      return nullptr;
    }

    char label[256];
    snprintf(label, sizeof(label), "Xenia Converted Index Buffer (%zu bytes)",
             allocation_size);
    new_buffer->setLabel(NS::String::string(label, NS::UTF8StringEncoding));

    frame_index_buffers_.push_back({new_buffer, allocation_size, 0});
    chosen_buffer = &frame_index_buffers_.back();

    XELOGI("Created new Metal index buffer for primitive conversion ({} bytes)",
           allocation_size);
  }

  // Mark buffer as used this frame
  chosen_buffer->last_frame_used = current_frame;

  // Return the buffer handle and CPU mapping.
  uint64_t gpu_offset = 0;
  void* cpu_buffer = chosen_buffer->buffer->contents();

  // Apply SIMD co-alignment if requested
  if (coalign_for_simd) {
    ptrdiff_t offset =
        GetSimdCoalignmentOffset(cpu_buffer, coalignment_original_address);
    cpu_buffer = static_cast<uint8_t*>(cpu_buffer) + offset;
    gpu_offset += uint64_t(offset);
  }

  backend_handle_out = converted_index_buffers_.size();
  converted_index_buffers_.push_back({chosen_buffer->buffer, gpu_offset});

  return cpu_buffer;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
