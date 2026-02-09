/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_function.h"

#include <atomic>
#include <cstring>

// pthread_jit_write_protect_np is only available on macOS ARM64.
// On iOS, the dual-mapping (split W^X via vm_remap) path is used instead,
// so we never need to toggle JIT write protection per-thread.
#if XE_PLATFORM_APPLE && !XE_PLATFORM_IOS
#include <pthread.h>
#endif

#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/thread_state.h"

#if XE_PLATFORM_APPLE && !XE_PLATFORM_IOS && defined(__aarch64__)
thread_local bool jit_thread_initialized = false;

// Initialize JIT execution for the current thread
static void InitializeJITThread() {
  if (!jit_thread_initialized) {
    // Ensure this thread can execute JIT code by setting execute mode
    pthread_jit_write_protect_np(1);  // Enable execute, disable write
    jit_thread_initialized = true;
  }
}
#endif

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

A64Function::A64Function(Module* module, uint32_t address)
    : GuestFunction(module, address) {}

A64Function::~A64Function() {
  // machine_code_ is freed by code cache.
}

void A64Function::Setup(uint8_t* machine_code, size_t machine_code_length) {
  machine_code_ = machine_code;
  machine_code_length_ = machine_code_length;
}

bool A64Function::CallImpl(ThreadState* thread_state, uint32_t return_address) {
#if XE_PLATFORM_APPLE && !XE_PLATFORM_IOS && defined(__aarch64__)
  // Initialize JIT execution for this thread (macOS only).
  // On iOS, the dual-mapping path means execute pages are always executable.
  InitializeJITThread();
#endif

  auto backend =
      reinterpret_cast<A64Backend*>(thread_state->processor()->backend());
  auto thunk = backend->host_to_guest_thunk();

#if XE_PLATFORM_IOS && defined(__aarch64__)
  size_t target_region_size = 0;
  xe::memory::PageAccess target_region_access =
      xe::memory::PageAccess::kNoAccess;
  const bool target_region_ok = xe::memory::QueryProtect(
      machine_code_, target_region_size, target_region_access);

  size_t thunk_region_size = 0;
  xe::memory::PageAccess thunk_region_access =
      xe::memory::PageAccess::kNoAccess;
  const bool thunk_region_ok = xe::memory::QueryProtect(
      reinterpret_cast<void*>(thunk), thunk_region_size, thunk_region_access);

  const bool target_executable =
      target_region_access == xe::memory::PageAccess::kExecuteReadOnly ||
      target_region_access == xe::memory::PageAccess::kExecuteReadWrite;
  const bool thunk_executable =
      thunk_region_access == xe::memory::PageAccess::kExecuteReadOnly ||
      thunk_region_access == xe::memory::PageAccess::kExecuteReadWrite;

  if (!target_region_ok || !target_executable || !thunk_region_ok ||
      !thunk_executable) {
    static std::atomic<bool> logged_bad_jit_path{false};
    bool expected = false;
    if (logged_bad_jit_path.compare_exchange_strong(
            expected, true, std::memory_order_relaxed)) {
      XELOGE(
          "A64 iOS JIT call blocked due to non-exec QueryProtect: "
          "thunk={:p} target={:p} return=0x{:08X} "
          "thunk_ok={} thunk_access={} thunk_size=0x{:X} "
          "target_ok={} target_access={} target_size=0x{:X}",
          reinterpret_cast<void*>(thunk), static_cast<void*>(machine_code_),
          return_address, thunk_region_ok,
          static_cast<uint32_t>(thunk_region_access),
          static_cast<uint32_t>(thunk_region_size), target_region_ok,
          static_cast<uint32_t>(target_region_access),
          static_cast<uint32_t>(target_region_size));
    }
    return false;
  }

  static std::atomic<bool> logged_first_call{false};
  bool expected = false;
  if (logged_first_call.compare_exchange_strong(expected, true,
                                                std::memory_order_relaxed)) {
    uint32_t thunk_insn = 0;
    uint32_t target_insn = 0;
    std::memcpy(&thunk_insn, reinterpret_cast<void*>(thunk),
                sizeof(thunk_insn));
    std::memcpy(&target_insn, machine_code_, sizeof(target_insn));
    XELOGW(
        "A64 iOS first call: thunk={:p} target={:p} return=0x{:08X} "
        "thunk_ok={} thunk_size=0x{:X} thunk_access={} "
        "target_ok={} target_size=0x{:X} target_access={} "
        "thunk_insn=0x{:08X} target_insn=0x{:08X}",
        reinterpret_cast<void*>(thunk), static_cast<void*>(machine_code_),
        return_address, thunk_region_ok,
        static_cast<uint32_t>(thunk_region_size),
        static_cast<uint32_t>(thunk_region_access), target_region_ok,
        static_cast<uint32_t>(target_region_size),
        static_cast<uint32_t>(target_region_access), thunk_insn, target_insn);
  }
#endif

  // Make the actual thunk call
  thunk(machine_code_, thread_state->context(),
        reinterpret_cast<void*>(uintptr_t(return_address)));

  return true;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
