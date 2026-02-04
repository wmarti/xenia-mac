/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_BACKEND_A64_A64_BACKEND_H_
#define XENIA_CPU_BACKEND_A64_A64_BACKEND_H_

#include <memory>

#include "xenia/base/bit_map.h"
#include "xenia/base/cvar.h"
#include "xenia/cpu/backend/backend.h"

DECLARE_int32(a64_extension_mask);
DECLARE_bool(a64_enable_host_guest_stack_synchronization);
DECLARE_int64(max_stackpoints);

namespace xe {
class Exception;
}  // namespace xe
namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

class A64CodeCache;

typedef void* (*HostToGuestThunk)(void* target, void* arg0, void* arg1);
typedef void* (*GuestToHostThunk)(void* target, void* arg0, void* arg1);
typedef void (*ResolveFunctionThunk)();
typedef void (*StackSyncThunk)();

struct A64BackendStackpoint {
  uint64_t host_sp;
  uint64_t host_fp;
  uint32_t guest_sp;
  uint32_t guest_return_address;
  uint32_t stack_size;
  uint32_t reserved;
};
static_assert(sizeof(A64BackendStackpoint) == 32,
              "A64BackendStackpoint must be 32 bytes");

struct A64BackendContext {
  A64BackendStackpoint* stackpoints = nullptr;
  uint64_t pending_stack_sync_sp = 0;
  uint64_t pending_stack_sync_fp = 0;
  uint64_t pending_stack_sync_target = 0;
  uint32_t current_stackpoint_depth = 0;
  uint32_t pending_stack_sync = 0;
};

class A64Backend : public Backend {
 public:
  static const uint32_t kForceReturnAddress = 0x9FFF0000u;
  // Guest trampoline range mirrors x64 to keep kernel expectations consistent.
  static constexpr uint32_t kGuestTrampolineBase = 0x80000000;
  static constexpr uint32_t kGuestTrampolineEnd = 0x80040000;
  static constexpr uint32_t kGuestTrampolineMinLen = 8;
  static constexpr uint32_t kMaxGuestTrampolines =
      (kGuestTrampolineEnd - kGuestTrampolineBase) /
      kGuestTrampolineMinLen;

  explicit A64Backend();
  ~A64Backend() override;

  A64CodeCache* code_cache() const { return code_cache_.get(); }
  uintptr_t emitter_data() const { return emitter_data_; }

  // Call a generated function, saving all stack parameters.
  HostToGuestThunk host_to_guest_thunk() const { return host_to_guest_thunk_; }
  // Function that guest code can call to transition into host code.
  GuestToHostThunk guest_to_host_thunk() const { return guest_to_host_thunk_; }
  // Function that thunks to the ResolveFunction in A64Emitter.
  ResolveFunctionThunk resolve_function_thunk() const {
    return resolve_function_thunk_;
  }
  StackSyncThunk stack_sync_thunk() const { return stack_sync_thunk_; }
  StackSyncThunk stack_sync_helper() const { return stack_sync_helper_; }

  bool Initialize(Processor* processor) override;

  void CommitExecutableRange(uint32_t guest_low, uint32_t guest_high) override;

  std::unique_ptr<Assembler> CreateAssembler() override;

  std::unique_ptr<GuestFunction> CreateGuestFunction(Module* module,
                                                     uint32_t address) override;

  uint64_t CalculateNextHostInstruction(ThreadDebugInfo* thread_info,
                                        uint64_t current_pc) override;

  void InstallBreakpoint(Breakpoint* breakpoint) override;
  void InstallBreakpoint(Breakpoint* breakpoint, Function* fn) override;
  void UninstallBreakpoint(Breakpoint* breakpoint) override;
  void RecordMMIOExceptionForGuestInstruction(void* host_address);
  void InitializeBackendContext(void* ctx) override;
  void DeinitializeBackendContext(void* ctx) override;
  void PrepareForReentry(void* ctx) override;
  void SetGuestRoundingMode(void* ctx, unsigned int mode) override;
  uint32_t CreateGuestTrampoline(GuestTrampolineProc proc, void* userdata1,
                                 void* userdata2,
                                 bool long_term = false) override;
  void FreeGuestTrampoline(uint32_t trampoline_addr) override;
  A64BackendContext* BackendContextForGuestContext(void* ctx) {
    return reinterpret_cast<A64BackendContext*>(
        reinterpret_cast<intptr_t>(ctx) - sizeof(A64BackendContext));
  }

 private:
  static bool ExceptionCallbackThunk(Exception* ex, void* data);
  bool ExceptionCallback(Exception* ex);

  uintptr_t capstone_handle_ = 0;

  std::unique_ptr<A64CodeCache> code_cache_;
  uintptr_t emitter_data_ = 0;

  HostToGuestThunk host_to_guest_thunk_;
  GuestToHostThunk guest_to_host_thunk_;
  ResolveFunctionThunk resolve_function_thunk_;
  StackSyncThunk stack_sync_thunk_ = nullptr;
  StackSyncThunk stack_sync_helper_ = nullptr;

  uint8_t* guest_trampoline_memory_ = nullptr;
  BitMap guest_trampoline_address_bitmap_;
};

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_BACKEND_H_
