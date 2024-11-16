// a64_code_cache_macos.cc
/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.
 * Released under the BSD license - see LICENSE in the root for more details.
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_code_cache.h"

#include <atomic>
#include <cstring>
#include <vector>
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <mutex>
#include <unistd.h> // Added for getpid()

#include "xenia/base/assert.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/function_debug_info.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

// Size of unwind info per function
static constexpr size_t kUnwindInfoSize = 32;  // Increased to accommodate DWARF unwind info

std::unique_ptr<A64CodeCache> A64CodeCache::Create() {
  return std::make_unique<MacOSA64CodeCache>();
}

MacOSA64CodeCache::MacOSA64CodeCache() = default;

MacOSA64CodeCache::~MacOSA64CodeCache() {
  if (trampoline_table_base_) {
    xe::memory::DeallocFixed(trampoline_table_base_, kTrampolineTableSize,
                            xe::memory::DeallocationType::kRelease);
  }
}

bool MacOSA64CodeCache::Initialize() {
  const size_t page_size = xe::memory::page_size();

  // Initialize trampoline table first
  if (!InitializeTrampolineTable()) {
    XELOGE("Failed to initialize trampoline table");
    return false;
  }

  // Let macOS choose the address for the indirection table, but ensure size is page aligned
  const size_t aligned_indirection_size = (kIndirectionTableSize + (page_size-1)) & ~(page_size-1);
  indirection_table_base_ = reinterpret_cast<uint8_t*>(xe::memory::AllocFixed(
      nullptr,
      aligned_indirection_size,
      xe::memory::AllocationType::kReserve,
      xe::memory::PageAccess::kReadWrite));

  if (!indirection_table_base_) {
    XELOGE("Unable to allocate code cache indirection table");
    return false;
  }

  // Commit initial page
  if (!xe::memory::Protect(indirection_table_base_, page_size,
                          xe::memory::PageAccess::kReadWrite)) {
    XELOGE("Unable to commit indirection table memory");
    xe::memory::DeallocFixed(indirection_table_base_, aligned_indirection_size,
                            xe::memory::DeallocationType::kRelease);
    return false;
  }

  // Create mmap file for executable code
  std::string code_cache_file = fmt::format("xenia_code_cache_{}", getpid());
  const size_t aligned_code_size = (kGeneratedCodeSize + (page_size-1)) & ~(page_size-1);
  mapping_ = xe::memory::CreateFileMappingHandle(
      code_cache_file, aligned_code_size, xe::memory::PageAccess::kReadWrite,
      false);
  if (mapping_ == xe::memory::kFileMappingHandleInvalid) {
    XELOGE("Unable to create code cache mmap");
    return false;
  }

  // Allocate code cache with proper alignment and spacing
  uint64_t code_cache_base = reinterpret_cast<uint64_t>(trampoline_table_base_) + kTrampolineTableSize;
  code_cache_base = (code_cache_base + (page_size-1)) & ~(page_size-1);  // Align to page boundary

  // Map execute view first at base address
  generated_code_execute_base_ =
      reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
          mapping_, reinterpret_cast<void*>(code_cache_base), aligned_code_size,
          xe::memory::PageAccess::kExecuteReadOnly, 0));
  if (!generated_code_execute_base_) {
    XELOGE("Failed to allocate code cache execute view at 0x{:X}", code_cache_base);
    return false;
  }

  // Map write view at offset
  uint64_t write_base = code_cache_base + aligned_code_size;
  write_base = (write_base + (page_size-1)) & ~(page_size-1);  // Ensure alignment
  generated_code_write_base_ =
      reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
          mapping_, reinterpret_cast<void*>(write_base), aligned_code_size,
          xe::memory::PageAccess::kReadWrite, 0));
  if (!generated_code_write_base_) {
    XELOGE("Failed to allocate code cache write view at 0x{:X}", write_base);
    xe::memory::UnmapFileView(mapping_, generated_code_execute_base_, aligned_code_size);
    return false;
  }

  // Pre-commit initial block of memory with proper permissions
  size_t initial_commit_size = 4 * 1024 * 1024;  // 4MB initial commit
  initial_commit_size = (initial_commit_size + (page_size-1)) & ~(page_size-1);

  // Commit write view first
  if (!xe::memory::AllocFixed(generated_code_write_base_, initial_commit_size,
                             xe::memory::AllocationType::kCommit,
                             xe::memory::PageAccess::kReadWrite)) {
    XELOGE("Failed to commit initial code cache memory (write)");
    return false;
  }

  // Then commit execute view
  if (!xe::memory::AllocFixed(generated_code_execute_base_, initial_commit_size,
                             xe::memory::AllocationType::kCommit,
                             xe::memory::PageAccess::kExecuteReadOnly)) {
    XELOGE("Failed to commit initial code cache memory (execute)");
    return false;
  }

  // Initialize offset to account for committed memory
  generated_code_offset_ = 0;
  generated_code_commit_mark_ = initial_commit_size;

  // Pre-allocate unwind table to avoid resizing
  unwind_table_.resize(kMaximumFunctionCount);
  unwind_table_count_ = 0;

  // Store thunk addresses from execute view with proper offsets
  uint32_t thunk_offset = 0;
  host_to_guest_thunk_ = reinterpret_cast<HostToGuestThunk>(generated_code_execute_base_ + thunk_offset);
  thunk_offset += 0x80;  // Space for thunk code
  guest_to_host_thunk_ = reinterpret_cast<GuestToHostThunk>(generated_code_execute_base_ + thunk_offset);
  thunk_offset += 0x80;  // Space for thunk code
  resolve_function_thunk_ = reinterpret_cast<ResolveFunctionThunk>(generated_code_execute_base_ + thunk_offset);

  // Ensure initial memory is properly synchronized
  std::atomic_thread_fence(std::memory_order_release);
  __builtin___clear_cache(
      reinterpret_cast<char*>(generated_code_write_base_),
      reinterpret_cast<char*>(generated_code_write_base_ + initial_commit_size));

  XELOGI("Successfully allocated code cache at 0x{:X} (write: 0x{:X}, execute: 0x{:X}, page_size: 0x{:X})",
         code_cache_base,
         reinterpret_cast<uint64_t>(generated_code_write_base_),
         reinterpret_cast<uint64_t>(generated_code_execute_base_),
         page_size);

  return true;
}

bool MacOSA64CodeCache::InitializeTrampolineTable() {
  const size_t page_size = xe::memory::page_size();
  const size_t alloc_size = kTrampolineTableSize;  // Use full 16MB allocation
  
  // Create a file mapping for the trampoline table with RW permissions initially
  std::string trampoline_file = fmt::format("xenia_trampoline_{}", getpid());
  mapping_trampoline_ = xe::memory::CreateFileMappingHandle(
      trampoline_file, alloc_size,
      xe::memory::PageAccess::kReadWrite, false);
  if (mapping_trampoline_ == xe::memory::kFileMappingHandleInvalid) {
    XELOGE("Unable to create trampoline table mapping");
    return false;
  }

  // Map execute view first at the lower address
  uint64_t exec_address = 0x420000000;  // Fixed execute view address
  exec_address = xe::round_up(exec_address, page_size);

  trampoline_table_base_ = reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
      mapping_trampoline_, reinterpret_cast<void*>(exec_address),
      alloc_size, xe::memory::PageAccess::kExecuteReadOnly, 0));

  if (!trampoline_table_base_) {
    XELOGE("Unable to map trampoline table execute view at 0x{:X}", exec_address);
    xe::memory::CloseFileMappingHandle(mapping_trampoline_, trampoline_file);
    return false;
  }

  // Map write view at a higher address
  uint64_t write_address = 0x430000000;  // Fixed write view address
  write_address = xe::round_up(write_address, page_size);
  
  trampoline_table_write_base_ = reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
      mapping_trampoline_, reinterpret_cast<void*>(write_address),
      alloc_size, xe::memory::PageAccess::kReadWrite, 0));

  if (!trampoline_table_write_base_) {
    XELOGE("Unable to map trampoline table write view at 0x{:X}", write_address);
    xe::memory::UnmapFileView(mapping_trampoline_, trampoline_table_base_, alloc_size);
    xe::memory::CloseFileMappingHandle(mapping_trampoline_, trampoline_file);
    return false;
  }

  // Pre-commit initial block of memory
  size_t initial_commit_size = 64 * 1024;  // 64KB initial commit
  initial_commit_size = xe::round_up(initial_commit_size, page_size);

  // Commit write view first
  if (!xe::memory::AllocFixed(trampoline_table_write_base_, initial_commit_size,
                             xe::memory::AllocationType::kCommit,
                             xe::memory::PageAccess::kReadWrite)) {
    XELOGE("Failed to commit initial trampoline table memory (write)");
    return false;
  }

  // Then commit execute view
  if (!xe::memory::AllocFixed(trampoline_table_base_, initial_commit_size,
                             xe::memory::AllocationType::kCommit,
                             xe::memory::PageAccess::kExecuteReadOnly)) {
    XELOGE("Failed to commit initial trampoline table memory (execute)");
    return false;
  }

  // Initialize offset
  trampoline_table_offset_ = 0;

  // Pre-allocate trampoline entries vector to avoid resizing
  trampoline_entries_.reserve(kMaximumFunctionCount);

  // Ensure initial memory is properly synchronized
  std::atomic_thread_fence(std::memory_order_release);
  __builtin___clear_cache(
      reinterpret_cast<char*>(trampoline_table_write_base_),
      reinterpret_cast<char*>(trampoline_table_write_base_ + initial_commit_size));

  XELOGI("Successfully allocated trampoline table at 0x{:X} (write: 0x{:X})",
         reinterpret_cast<uint64_t>(trampoline_table_base_),
         reinterpret_cast<uint64_t>(trampoline_table_write_base_));

  return true;
}

A64CodeCache::UnwindReservation MacOSA64CodeCache::RequestUnwindReservation(uint8_t* entry_address) {
  A64CodeCache::UnwindReservation unwind_reservation;
  unwind_reservation.data_size = xe::round_up(kUnwindInfoSize, 16);
  
  // Get next slot atomically
  uint32_t slot = unwind_table_count_.fetch_add(1);
  if (slot >= kMaximumFunctionCount) {
    XELOGE("Exceeded maximum function count ({})", kMaximumFunctionCount);
    unwind_reservation.table_slot = 0;  // Use first slot as fallback
    return unwind_reservation;
  }
  
  unwind_reservation.table_slot = slot;
  unwind_reservation.entry_address = entry_address;
  return unwind_reservation;
}

void MacOSA64CodeCache::PlaceCode(uint32_t guest_address, void* machine_code,
                               const EmitFunctionInfo& func_info,
                               void* code_execute_address,
                               A64CodeCache::UnwindReservation unwind_reservation) {
  XELOGI("=== PlaceCode Debug Info ===");
  XELOGI("  Guest address: {:#x}", guest_address);
  XELOGI("  Execute address: {}", fmt::ptr(code_execute_address));
  XELOGI("  Code size: {:#x} bytes", func_info.code_size.total);

  // Add unwind info if we have a valid slot
  if (unwind_reservation.table_slot < kMaximumFunctionCount) {
    UnwindInfo& unwind_info = unwind_table_[unwind_reservation.table_slot];
    unwind_info.begin_address = reinterpret_cast<uintptr_t>(code_execute_address);
    unwind_info.end_address = unwind_info.begin_address + func_info.code_size.total;
    unwind_info.offset = 0;  // Not used on macOS
    unwind_info.size = static_cast<uint32_t>(func_info.code_size.total);
  }

  // Calculate write address from execute address
  uint8_t* write_address = generated_code_write_base_ + 
      (reinterpret_cast<uint8_t*>(code_execute_address) - generated_code_execute_base_);

  XELOGI("  Write address: {}", fmt::ptr(write_address));

  // Copy code to write view
  std::memcpy(write_address, machine_code, func_info.code_size.total);

  // Ensure memory writes are visible
  std::atomic_thread_fence(std::memory_order_release);

  // Flush instruction cache to ensure code is visible
  constexpr size_t kCacheLineSize = 16;  // ARM64 typical cache line size
  uint8_t* start_addr = write_address;
  uint8_t* end_addr = write_address + func_info.code_size.total;

  XELOGI("  Cache flush range: {} to {}", fmt::ptr(start_addr), fmt::ptr(end_addr));
  XELOGI("  Size: {} bytes", end_addr - start_addr);

  __builtin___clear_cache(reinterpret_cast<char*>(start_addr),
                         reinterpret_cast<char*>(end_addr));

  // If this is a thunk that needs to be in low memory, create a trampoline
  if (guest_address && indirection_table_base_) {
    uint32_t* indirection_slot = reinterpret_cast<uint32_t*>(
        indirection_table_base_ + (guest_address - kIndirectionTableBase));
    
    // Check if target is within direct branch range
    int64_t distance = static_cast<int64_t>(reinterpret_cast<uint64_t>(code_execute_address)) -
                      static_cast<int64_t>(reinterpret_cast<uint64_t>(indirection_slot));
    
    constexpr int64_t kMaxBranchDistance = 128 * 1024 * 1024;  // 128MB
    bool needs_trampoline = std::abs(distance) > kMaxBranchDistance;

    XELOGI("  Distance to target: {:#x} bytes", distance);
    XELOGI("  Needs trampoline: {}", needs_trampoline);

    if (needs_trampoline) {
      void* trampoline = AllocateTrampoline(code_execute_address);
      if (trampoline) {
        XELOGI("  Created trampoline at {} for target {}", 
               fmt::ptr(trampoline), fmt::ptr(code_execute_address));
        *indirection_slot = static_cast<uint32_t>(
            reinterpret_cast<uint64_t>(trampoline));
      } else {
        XELOGE("Failed to allocate trampoline for guest address {:#x}", guest_address);
        *indirection_slot = static_cast<uint32_t>(
            reinterpret_cast<uint64_t>(code_execute_address));
      }
    } else {
      XELOGI("  Using direct branch to target");
      *indirection_slot = static_cast<uint32_t>(
          reinterpret_cast<uint64_t>(code_execute_address));
    }
  }

  XELOGI("=== End PlaceCode Debug Info ===");
}

void* MacOSA64CodeCache::CreateTrampoline(void* target_address) {
  if (!target_address) {
    XELOGE("CreateTrampoline: target_address is null");
    return nullptr;
  }

  if (!trampoline_table_base_ || !trampoline_table_write_base_) {
    XELOGE("CreateTrampoline: trampoline table not initialized");
    return nullptr;
  }

  XELOGI("=== CreateTrampoline Debug Info ===");
  XELOGI("Memory Layout:");
  XELOGI("  Trampoline table base (exec): {}", fmt::ptr(trampoline_table_base_));
  XELOGI("  Trampoline table base (write): {}", fmt::ptr(trampoline_table_write_base_));
  XELOGI("  Code cache base (exec): {}", fmt::ptr(generated_code_execute_base_));
  XELOGI("  Code cache base (write): {}", fmt::ptr(generated_code_write_base_));
  
  // Calculate aligned offset for new trampoline
  size_t aligned_offset = (trampoline_table_offset_.load() + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
  if (aligned_offset + kTrampolineSize > kTrampolineTableSize) {
    XELOGE("Trampoline table full!");
    return nullptr;
  }

  uint8_t* write_ptr = trampoline_table_write_base_ + aligned_offset;
  uint8_t* exec_ptr = trampoline_table_base_ + aligned_offset;

  XELOGI("Trampoline Details:");
  XELOGI("  Aligned offset: {:#x}", aligned_offset);
  XELOGI("  Write pointer: {}", fmt::ptr(write_ptr));
  XELOGI("  Execute pointer: {}", fmt::ptr(exec_ptr));
  XELOGI("  Target address: {}", fmt::ptr(target_address));

  // Calculate branch
  uint64_t target = reinterpret_cast<uint64_t>(target_address);
  uint64_t source = reinterpret_cast<uint64_t>(exec_ptr);
  int64_t distance = static_cast<int64_t>(target) - static_cast<int64_t>(source);

  XELOGI("Branch Calculation:");
  XELOGI("  Source: {} ({:#x})", fmt::ptr((void*)source), source);
  XELOGI("  Target: {} ({:#x})", fmt::ptr((void*)target), target);
  XELOGI("  Distance: {:#x} ({} bytes)", distance, distance);

  // Check if branch distance is within range (±128MB)
  constexpr int64_t kMaxBranchDistance = 128 * 1024 * 1024;  // 128MB
  if (std::abs(distance) > kMaxBranchDistance) {
    XELOGE("Branch distance too far: {} bytes (max ±128MB)", distance);
    return nullptr;
  }

  // Generate branch instruction
  uint32_t branch_instruction = 0x14000000 | ((static_cast<uint32_t>(distance >> 2)) & 0x3FFFFFF);

  // Write branch instruction
  *reinterpret_cast<uint32_t*>(write_ptr) = branch_instruction;

  // Ensure write is visible
  std::atomic_thread_fence(std::memory_order_release);

  // Align memory range to cache line boundaries for flush
  uint8_t* flush_start = write_ptr;
  uint8_t* flush_end = write_ptr + kTrampolineSize;

  XELOGI("Cache Flush Range:");
  XELOGI("  Start: {}", fmt::ptr(flush_start));
  XELOGI("  End: {}", fmt::ptr(flush_end));
  XELOGI("  Size: {} bytes", flush_end - flush_start);

  __builtin___clear_cache(reinterpret_cast<char*>(flush_start),
                         reinterpret_cast<char*>(flush_end));

  // Update trampoline table offset
  trampoline_table_offset_.store(aligned_offset + kTrampolineSize, std::memory_order_release);

  XELOGI("Successfully created trampoline at {} to {}", 
         fmt::ptr(exec_ptr), fmt::ptr(target_address));
  XELOGI("=== End CreateTrampoline Debug Info ===");

  return exec_ptr;
}

void* MacOSA64CodeCache::AllocateTrampoline(void* target_address) {
  if (!target_address) {
    XELOGE("AllocateTrampoline: target_address is null");
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(trampoline_mutex_);

  XELOGI("=== AllocateTrampoline Debug Info ===");
  XELOGI("Target address: {}", fmt::ptr(target_address));

  // Check if we already have a trampoline for this target
  for (const auto& entry : trampoline_entries_) {
    if (entry.target_address == target_address) {
      XELOGI("Found existing trampoline at {} for target {}", 
             fmt::ptr(entry.trampoline_address), fmt::ptr(target_address));
      return entry.trampoline_address;
    }
  }

  // Create new trampoline
  void* trampoline = CreateTrampoline(target_address);
  if (!trampoline) {
    XELOGE("Failed to create trampoline for target {}", fmt::ptr(target_address));
    return nullptr;
  }

  // Store trampoline entry for future reference
  TrampolineEntry entry;
  entry.target_address = target_address;
  entry.trampoline_address = reinterpret_cast<uint8_t*>(trampoline);
  trampoline_entries_.push_back(entry);

  XELOGI("Successfully allocated trampoline at {} for target {}", 
         fmt::ptr(trampoline), fmt::ptr(target_address));
  XELOGI("=== End AllocateTrampoline Debug Info ===");

  return trampoline;
}

void* MacOSA64CodeCache::LookupUnwindInfo(uint64_t host_pc) {
  // Binary search through unwind table
  uint32_t count = unwind_table_count_.load();
  for (uint32_t i = 0; i < count; ++i) {
    const UnwindInfo& info = unwind_table_[i];
    if (host_pc < info.begin_address) {
      continue;
    } else if (host_pc >= info.end_address) {
      continue;
    }
    return reinterpret_cast<void*>(info.begin_address);
  }
  return nullptr;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
