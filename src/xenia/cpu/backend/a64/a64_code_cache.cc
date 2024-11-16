/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_code_cache.h"

#include <cstdlib>
#include <cstring>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/assert.h"
#include "xenia/base/clock.h"
#include "xenia/base/literals.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/module.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

using namespace xe::literals;

A64CodeCache::A64CodeCache() = default;

A64CodeCache::~A64CodeCache() {
  if (indirection_table_base_) {
    xe::memory::DeallocFixed(indirection_table_base_, 0,
                             xe::memory::DeallocationType::kRelease);
  }

  // Unmap all views and close mapping.
  if (mapping_ != xe::memory::kFileMappingHandleInvalid) {
    if (generated_code_write_base_ &&
        generated_code_write_base_ != generated_code_execute_base_) {
      xe::memory::UnmapFileView(mapping_, generated_code_write_base_,
                                kGeneratedCodeSize);
    }
    if (generated_code_execute_base_) {
      xe::memory::UnmapFileView(mapping_, generated_code_execute_base_,
                                kGeneratedCodeSize);
    }
    xe::memory::CloseFileMappingHandle(mapping_, file_name_);
    mapping_ = xe::memory::kFileMappingHandleInvalid;
  }
}

bool A64CodeCache::Initialize() {
  indirection_table_base_ = reinterpret_cast<uint8_t*>(xe::memory::AllocFixed(
      reinterpret_cast<void*>(kIndirectionTableBase), kIndirectionTableSize,
      xe::memory::AllocationType::kReserve,
      xe::memory::PageAccess::kReadWrite));
  if (!indirection_table_base_) {
    XELOGE("Unable to allocate code cache indirection table");
    XELOGE(
        "This is likely because the {:X}-{:X} range is in use by some other "
        "system DLL",
        static_cast<uint64_t>(kIndirectionTableBase),
        kIndirectionTableBase + kIndirectionTableSize);
  }

  // Create mmap file. This allows us to share the code cache with the debugger.
  file_name_ = fmt::format("xenia_code_cache");
  mapping_ = xe::memory::CreateFileMappingHandle(
      file_name_, kGeneratedCodeSize, xe::memory::PageAccess::kExecuteReadWrite,
      false);
  if (mapping_ == xe::memory::kFileMappingHandleInvalid) {
    XELOGE("Unable to create code cache mmap");
    return false;
  }

  // Map generated code region into the file. Pages are committed as required.
  if (xe::memory::IsWritableExecutableMemoryPreferred()) {
    generated_code_execute_base_ =
        reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
            mapping_, reinterpret_cast<void*>(kGeneratedCodeExecuteBase),
            kGeneratedCodeSize, xe::memory::PageAccess::kExecuteReadWrite, 0));
    generated_code_write_base_ = generated_code_execute_base_;
    if (!generated_code_execute_base_ || !generated_code_write_base_) {
      XELOGE("Unable to allocate code cache generated code storage");
      XELOGE(
          "This is likely because the {:X}-{:X} range is in use by some other "
          "system DLL",
          uint64_t(kGeneratedCodeExecuteBase),
          uint64_t(kGeneratedCodeExecuteBase + kGeneratedCodeSize));
      return false;
    }
  } else {
    generated_code_execute_base_ =
        reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
            mapping_, reinterpret_cast<void*>(kGeneratedCodeExecuteBase),
            kGeneratedCodeSize, xe::memory::PageAccess::kExecuteReadOnly, 0));
    generated_code_write_base_ =
        reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
            mapping_, reinterpret_cast<void*>(kGeneratedCodeWriteBase),
            kGeneratedCodeSize, xe::memory::PageAccess::kReadWrite, 0));
    if (!generated_code_execute_base_ || !generated_code_write_base_) {
      XELOGE("Unable to allocate code cache generated code storage");
      XELOGE(
          "This is likely because the {:X}-{:X} and {:X}-{:X} ranges are in "
          "use by some other system DLL",
          uint64_t(kGeneratedCodeExecuteBase),
          uint64_t(kGeneratedCodeExecuteBase + kGeneratedCodeSize),
          uint64_t(kGeneratedCodeWriteBase),
          uint64_t(kGeneratedCodeWriteBase + kGeneratedCodeSize));
      return false;
    }
  }

  // Preallocate the function map to a large, reasonable size.
  generated_code_map_.reserve(kMaximumFunctionCount);

  return true;
}

void A64CodeCache::set_indirection_default(uint32_t default_value) {
  indirection_default_value_ = default_value;
}

void A64CodeCache::AddIndirection(uint32_t guest_address,
                                uint64_t host_address) {
  if (!indirection_table_base_) {
    return;
  }

  uint32_t* indirection_slot = reinterpret_cast<uint32_t*>(
      indirection_table_base_ + (guest_address - kIndirectionTableBase));

  // Calculate relative offset from the indirection slot to target
  int64_t offset = static_cast<int64_t>(host_address) - 
                   reinterpret_cast<int64_t>(indirection_slot);
                   
  // We need to store a branch instruction that can reach the target
  uint32_t branch_instruction;
  if (std::abs(offset) < (1LL << 27)) {  // ±128MB range for B
    // Direct branch: B target
    branch_instruction = 0x14000000 | ((offset >> 2) & 0x3FFFFFF);
  } else {
    XELOGE("Target address too far for branch instruction: offset = {}", offset);
    // Fall back to a safe default that will trap if executed
    branch_instruction = 0xD4200000;  // BRK #0
  }
  
  *indirection_slot = branch_instruction;
}

void A64CodeCache::CommitExecutableRange(uint32_t guest_low,
                                         uint32_t guest_high) {
  if (!indirection_table_base_) {
    return;
  }

  // Commit the memory.
  xe::memory::AllocFixed(
      indirection_table_base_ + (guest_low - kIndirectionTableBase),
      guest_high - guest_low, xe::memory::AllocationType::kCommit,
      xe::memory::PageAccess::kReadWrite);

  // Fill memory with the default value.
  uint32_t* p = reinterpret_cast<uint32_t*>(indirection_table_base_);
  for (uint32_t address = guest_low; address < guest_high; ++address) {
    p[(address - kIndirectionTableBase) / 4] = indirection_default_value_;
  }
}

void A64CodeCache::PlaceHostCode(uint32_t guest_address, void* machine_code,
                                 const EmitFunctionInfo& func_info,
                                 void*& code_execute_address_out,
                                 void*& code_write_address_out) {
  // Same for now. We may use different pools or whatnot later on, like when
  // we only want to place guest code in a serialized cache on disk.
  PlaceGuestCode(guest_address, machine_code, func_info, nullptr,
                 code_execute_address_out, code_write_address_out);
}

void A64CodeCache::PlaceGuestCode(uint32_t guest_address, void* machine_code,
                                  const EmitFunctionInfo& func_info,
                                  GuestFunction* function_info,
                                  void*& code_execute_address_out,
                                  void*& code_write_address_out) {
  const size_t page_size = 16 * 1024;  // 16KB macOS ARM64 page size
  
  // Validate code size components
  size_t computed_total = func_info.code_size.prolog + 
                         func_info.code_size.body +
                         func_info.code_size.epilog + 
                         func_info.code_size.tail;
                         
  XELOGI("PlaceGuestCode: guest_address={:08X}, machine_code={:p}", guest_address, machine_code);
  XELOGI("  Code size details: prolog={:X}, body={:X}, epilog={:X}, tail={:X}", 
         func_info.code_size.prolog, func_info.code_size.body,
         func_info.code_size.epilog, func_info.code_size.tail);
  XELOGI("  Total sizes: computed={:X}, stored={:X}", 
         computed_total, func_info.code_size.total);
         
  if (computed_total != func_info.code_size.total) {
    XELOGE("Code size mismatch! Using computed total for safety");
    const_cast<EmitFunctionInfo&>(func_info).code_size.total = computed_total;
  }
  
  if (computed_total == 0) {
    XELOGE("Invalid code size - zero length code block");
    return;
  }

  // Calculate total size needed for code
  // ARM64 instructions must be 4-byte aligned, but we allocate in 16-byte chunks
  // for better memory alignment
  size_t code_size = xe::round_up(func_info.code_size.total, 16);
  generated_code_offset_ += code_size;
  XELOGI("  Code placement: actual_size={:X}, aligned_size={:X}, new_offset={:X}", 
         func_info.code_size.total, code_size, generated_code_offset_);

  // Calculate low/high marks for code memory
  size_t low_mark = generated_code_offset_ - code_size;
  size_t high_mark = generated_code_offset_;

  // Get both write and execute addresses for the same offset
  uint8_t* code_write_address = generated_code_write_base_ + low_mark;
  uint8_t* code_execute_address = generated_code_execute_base_ + low_mark;

  // Commit memory if needed
  size_t old_commit_mark = generated_code_commit_mark_.load();
  size_t new_commit_mark = old_commit_mark;
  if (high_mark > old_commit_mark) {
    new_commit_mark = xe::round_up(high_mark, page_size);
    XELOGI("  Committing memory: new_mark={:X}", new_commit_mark);
    
    if (!xe::memory::AllocFixed(
        generated_code_execute_base_ + old_commit_mark,
        new_commit_mark - old_commit_mark,
        xe::memory::AllocationType::kCommit,
        xe::memory::PageAccess::kExecuteReadOnly)) {
      XELOGE("Failed to commit execute memory at {:p} size {:X}", 
             generated_code_execute_base_ + old_commit_mark,
             new_commit_mark - old_commit_mark);
      return;
    }
    
    if (!xe::memory::AllocFixed(
        generated_code_write_base_ + old_commit_mark,
        new_commit_mark - old_commit_mark,
        xe::memory::AllocationType::kCommit,
        xe::memory::PageAccess::kReadWrite)) {
      XELOGE("Failed to commit write memory at {:p} size {:X}", 
             generated_code_write_base_ + old_commit_mark,
             new_commit_mark - old_commit_mark);
      return;
    }
    
    generated_code_commit_mark_.store(new_commit_mark);
  }

  // Copy code to write address
  XELOGI("  Copying code: src={:p}, dst={:p}, size={:X}", 
         machine_code, code_write_address, func_info.code_size.total);
         
  if (!code_write_address || !machine_code) {
    XELOGE("Invalid pointers for code copy: write={:p}, machine={:p}", 
           code_write_address, machine_code);
    return;
  }

  // Copy the actual code first
  std::memcpy(code_write_address, machine_code, func_info.code_size.total);

  // Zero the padding between actual code and aligned size
  size_t padding = code_size - func_info.code_size.total;
  if (padding > 0 && padding < 16) {  // Sanity check on padding
    std::memset(code_write_address + func_info.code_size.total, 0x00, padding);
  }

  // Return both addresses
  code_execute_address_out = code_execute_address;
  code_write_address_out = code_write_address;

  // Notify subclasses of placed code.
  UnwindReservation unwind_reservation;
  if (function_info) {
    unwind_reservation = RequestUnwindReservation(code_execute_address);
  }
  PlaceCode(guest_address, machine_code, func_info, code_execute_address,
            unwind_reservation);
}

uint32_t A64CodeCache::PlaceData(const void* data, size_t length) {
  // Hold a lock while we bump the pointers up.
  size_t high_mark;
  uint8_t* data_address = nullptr;
  {
    auto global_lock = global_critical_region_.Acquire();

    // Reserve code.
    // Always move the code to land on 16b alignment.
    data_address = generated_code_write_base_ + generated_code_offset_;
    generated_code_offset_ += xe::round_up(length, 16);

    high_mark = generated_code_offset_;
  }

  // If we are going above the high water mark of committed memory, commit some
  // more. It's ok if multiple threads do this, as redundant commits aren't
  // harmful.
  size_t old_commit_mark, new_commit_mark;
  do {
    old_commit_mark = generated_code_commit_mark_;
    if (high_mark <= old_commit_mark) break;

    new_commit_mark = old_commit_mark + 16_MiB;
    if (generated_code_execute_base_ == generated_code_write_base_) {
      xe::memory::AllocFixed(generated_code_execute_base_, new_commit_mark,
                             xe::memory::AllocationType::kCommit,
                             xe::memory::PageAccess::kExecuteReadWrite);
    } else {
      xe::memory::AllocFixed(generated_code_execute_base_, new_commit_mark,
                             xe::memory::AllocationType::kCommit,
                             xe::memory::PageAccess::kExecuteReadOnly);
      xe::memory::AllocFixed(generated_code_write_base_, new_commit_mark,
                             xe::memory::AllocationType::kCommit,
                             xe::memory::PageAccess::kReadWrite);
    }
  } while (generated_code_commit_mark_.compare_exchange_weak(old_commit_mark,
                                                             new_commit_mark));

  // Copy code.
  std::memcpy(data_address, data, length);

  return uint32_t(uintptr_t(data_address));
}

GuestFunction* A64CodeCache::LookupFunction(uint64_t host_pc) {
  uint32_t key = uint32_t(host_pc - kGeneratedCodeExecuteBase);
  void* fn_entry = std::bsearch(
      &key, generated_code_map_.data(), generated_code_map_.size() + 1,
      sizeof(std::pair<uint32_t, Function*>),
      [](const void* key_ptr, const void* element_ptr) {
        auto key = *reinterpret_cast<const uint32_t*>(key_ptr);
        auto element =
            reinterpret_cast<const std::pair<uint64_t, GuestFunction*>*>(
                element_ptr);
        if (key < (element->first >> 32)) {
          return -1;
        } else if (key > uint32_t(element->first)) {
          return 1;
        } else {
          return 0;
        }
      });
  if (fn_entry) {
    return reinterpret_cast<const std::pair<uint64_t, GuestFunction*>*>(
               fn_entry)
        ->second;
  } else {
    return nullptr;
  }
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
