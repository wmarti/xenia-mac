/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_code_cache.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>

#if XE_PLATFORM_APPLE
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#if XE_PLATFORM_APPLE && !XE_PLATFORM_IOS
#include <pthread.h>
#endif

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/assert.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
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

DEFINE_bool(a64_indirection_table_log, false,
            "Log A64 indirection table mapping and updates.", "CPU");
DEFINE_int32(a64_indirection_table_log_limit, 32,
             "Maximum number of A64 indirection table log entries.", "CPU");
DEFINE_bool(ios_jit_brk_prepare_fallback, true,
            "On iOS ARM64, if protection transitions fail, issue external JIT "
            "prepare breakpoint (brk #0xf00d/x16=1 by default; optional legacy "
            "brk #0x69) so external brokers can prepare the region, then retry.",
            "CPU");
DEFINE_bool(ios_jit_brk_use_universal_0xf00d, true,
            "On iOS ARM64, use universal JIT broker breakpoint brk #0xf00d "
            "(x16=1) instead of legacy brk #0x69 for external prepare. "
            "Enabled by default for modern StikDebug scripts.",
            "CPU");
DEFINE_int32(ios_jit_initial_external_prepare_bytes, 0,
             "On iOS ARM64 TXM startup, issue one external prepare for "
             "this many bytes from code cache base (0 means full code cache).",
             "CPU");
DEFINE_bool(ios_jit_non_txm_force_mprotect_flip, false,
            "On iOS ARM64, skip dual-map JIT setup and use single-view "
            "mprotect flips only (RW while writing, RX when published).",
            "CPU");

namespace {

bool ShouldLogIndirectionTable() {
  if (!cvars::a64_indirection_table_log) {
    return false;
  }
  const int32_t limit = cvars::a64_indirection_table_log_limit;
  if (limit <= 0) {
    return false;
  }
  static std::atomic<int32_t> log_count{0};
  const int32_t count = log_count.fetch_add(1, std::memory_order_relaxed);
  return count < limit;
}

#if XE_PLATFORM_IOS && XE_ARCH_ARM64
std::atomic<bool> ios_external_prepare_issued{false};
std::atomic<bool> ios_external_detach_issued{false};
std::atomic<bool> ios_force_universal_prepare_command{false};
constexpr uint32_t kLegacyPrepareRejectedResult = 0xE0000069u;
constexpr uint32_t kLegacyPrepareRejectedResultSwapped = 0x690000E0u;

std::string FindChildWithNameLength(const std::string& directory,
                                    size_t name_length) {
  DIR* dir = opendir(directory.c_str());
  if (!dir) {
    return std::string();
  }

  std::string found;
  while (dirent* entry = readdir(dir)) {
    const char* name = entry->d_name;
    if (!name || name[0] == '.') {
      continue;
    }
    if (std::strlen(name) == name_length) {
      found = directory + "/" + name;
      break;
    }
  }
  closedir(dir);
  return found;
}

bool IOSHasTXM() {
  static const bool has_txm = []() -> bool {
    if (const char* env = std::getenv("HAS_TXM")) {
      if (env[0] == '1' && env[1] == '\0') {
        return true;
      }
      if (env[0] == '0' && env[1] == '\0') {
        return false;
      }
    }

    const std::string preboot_uuid =
        FindChildWithNameLength("/System/Volumes/Preboot", 36);
    if (!preboot_uuid.empty()) {
      const std::string txm_root =
          FindChildWithNameLength(preboot_uuid + "/boot", 96);
      if (!txm_root.empty()) {
        const std::string txm_path =
            txm_root + "/usr/standalone/firmware/FUD/Ap,TrustedExecutionMonitor.img4";
        if (access(txm_path.c_str(), F_OK) == 0) {
          return true;
        }
      }
    }

    const std::string private_preboot_root =
        FindChildWithNameLength("/private/preboot", 96);
    if (!private_preboot_root.empty()) {
      const std::string txm_path = private_preboot_root +
                                   "/usr/standalone/firmware/FUD/"
                                   "Ap,TrustedExecutionMonitor.img4";
      if (access(txm_path.c_str(), F_OK) == 0) {
        return true;
      }
    }

    return false;
  }();
  return has_txm;
}

int IOSProductMajorVersion() {
  static const int major_version = []() -> int {
    size_t version_size = 0;
    if (sysctlbyname("kern.osproductversion", nullptr, &version_size, nullptr,
                     0) != 0 ||
        version_size == 0) {
      return -1;
    }

    std::string version(version_size, '\0');
    if (sysctlbyname("kern.osproductversion", version.data(), &version_size,
                     nullptr, 0) != 0 ||
        version_size == 0) {
      return -1;
    }
    if (!version.empty() && version.back() == '\0') {
      version.pop_back();
    }
    if (version.empty()) {
      return -1;
    }

    int parsed_major = 0;
    size_t index = 0;
    while (index < version.size() &&
           version[index] >= '0' &&
           version[index] <= '9') {
      parsed_major = parsed_major * 10 + (version[index] - '0');
      ++index;
    }
    return parsed_major > 0 ? parsed_major : -1;
  }();
  return major_version;
}

bool IOSUseTXMBrokerPath() {
  if (!IOSHasTXM()) {
    return false;
  }
  const int ios_major_version = IOSProductMajorVersion();
  return ios_major_version >= 26;
}

bool ShouldUseUniversalPrepareCommand() {
  return cvars::ios_jit_brk_use_universal_0xf00d ||
         ios_force_universal_prepare_command.load(std::memory_order_relaxed);
}

const char* ExternalPrepareBreakpointDescription() {
  return ShouldUseUniversalPrepareCommand() ? "brk #0xf00d (x16=1)"
                                            : "brk #0x69";
}

bool GetPageAlignedRange(void* address, size_t length, uintptr_t& aligned_start,
                         size_t& aligned_length) {
  if (!length) {
    aligned_start = 0;
    aligned_length = 0;
    return true;
  }
  const uintptr_t start = reinterpret_cast<uintptr_t>(address);
  const size_t page_size = xe::memory::page_size();
  aligned_start = start & ~(page_size - 1);
  const uintptr_t aligned_end = xe::align(start + length, page_size);
  if (aligned_end <= aligned_start) {
    aligned_length = 0;
    return true;
  }
  aligned_length = aligned_end - aligned_start;
  return true;
}

bool SetPageAlignedAccess(void* address, size_t length,
                          xe::memory::PageAccess access) {
  uintptr_t aligned_start = 0;
  size_t aligned_length = 0;
  if (!GetPageAlignedRange(address, length, aligned_start, aligned_length) ||
      !aligned_length) {
    return true;
  }
  return xe::memory::Protect(reinterpret_cast<void*>(aligned_start),
                             aligned_length, access);
}

bool SetPageAlignedAccessWithMaxProtRetry(void* address, size_t length,
                                          xe::memory::PageAccess access) {
  if (SetPageAlignedAccess(address, length, access)) {
    return true;
  }

  uintptr_t aligned_start = 0;
  size_t aligned_length = 0;
  if (!GetPageAlignedRange(address, length, aligned_start, aligned_length) ||
      !aligned_length) {
    return true;
  }

  vm_prot_t target_prot = 0;
  switch (access) {
    case xe::memory::PageAccess::kNoAccess:
      target_prot = VM_PROT_NONE;
      break;
    case xe::memory::PageAccess::kReadOnly:
      target_prot = VM_PROT_READ;
      break;
    case xe::memory::PageAccess::kReadWrite:
      target_prot = VM_PROT_READ | VM_PROT_WRITE;
      break;
    case xe::memory::PageAccess::kExecuteReadOnly:
      target_prot = VM_PROT_READ | VM_PROT_EXECUTE;
      break;
    case xe::memory::PageAccess::kExecuteReadWrite:
      target_prot = VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
      break;
    default:
      return false;
  }

  constexpr vm_prot_t kMaxProt = VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
  const kern_return_t kr_max =
      vm_protect(mach_task_self(), static_cast<vm_address_t>(aligned_start),
                 aligned_length, TRUE, kMaxProt);
  if (kr_max != KERN_SUCCESS) {
    XELOGE(
        "iOS JIT mprotect fallback: vm_protect set-max failed "
        "addr=0x{:X} len=0x{:X} kr={}",
        aligned_start, static_cast<uint32_t>(aligned_length), kr_max);
    return false;
  }

  const kern_return_t kr_set =
      vm_protect(mach_task_self(), static_cast<vm_address_t>(aligned_start),
                 aligned_length, FALSE, target_prot);
  if (kr_set != KERN_SUCCESS) {
    XELOGE(
        "iOS JIT mprotect fallback: vm_protect set-current failed "
        "addr=0x{:X} len=0x{:X} target=0x{:X} kr={}",
        aligned_start, static_cast<uint32_t>(aligned_length),
        static_cast<uint32_t>(target_prot), kr_set);
    return false;
  }
  return true;
}

#if defined(__aarch64__)
void InvokeUniversalPrepareBreakpoint(uintptr_t aligned_start,
                                      size_t aligned_length) {
  register uint64_t x0 __asm("x0") = static_cast<uint64_t>(aligned_start);
  register uint64_t x1 __asm("x1") = static_cast<uint64_t>(aligned_length);
  register uint64_t x16 __asm("x16") = 1;
  asm volatile("brk #0xf00d" : "+r"(x0), "+r"(x1), "+r"(x16) : : "memory");
}

void InvokeUniversalDetachBreakpoint() {
  register uint64_t x16 __asm("x16") = 0;
  asm volatile("brk #0xf00d" : "+r"(x16) : : "memory");
}
#endif

bool MaybeRequestExternalJitDetach() {
  if (!ShouldUseUniversalPrepareCommand()) {
    return true;
  }
  bool expected = false;
  if (!ios_external_detach_issued.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return true;
  }
#if defined(__aarch64__)
  // The universal StikDebug script supports command 0 (detach). Once the JIT
  // region has been prepared, detach so the broker doesn't sit in a signal
  // loop for the rest of process lifetime.
  InvokeUniversalDetachBreakpoint();
  return true;
#else
  return false;
#endif
}

bool RequestExternalJitPrepare(void* address, size_t length) {
  uintptr_t aligned_start = 0;
  size_t aligned_length = 0;
  if (!GetPageAlignedRange(address, length, aligned_start, aligned_length) ||
      !aligned_length) {
    return true;
  }

#if defined(__aarch64__)
  if (ShouldUseUniversalPrepareCommand()) {
    // StikDebug universal JIT script path:
    // x16=1 + brk #0xf00d => prepare region (x0=addr, x1=len).
    InvokeUniversalPrepareBreakpoint(aligned_start, aligned_length);
    MaybeRequestExternalJitDetach();
    return true;
  }

  register uint64_t x0 __asm("x0") = static_cast<uint64_t>(aligned_start);
  register uint64_t x1 __asm("x1") = static_cast<uint64_t>(aligned_length);
  // Legacy broker path expected by older StikDebug scripts:
  // brk #0x69 with x0=addr, x1=len.
  asm volatile("brk #0x69" : "+r"(x0), "+r"(x1) : : "memory");

  // Newer StikDebug scripts intentionally reject legacy 0x69 and ask callers
  // to migrate to universal 0xf00d command dispatch.
  const uint32_t legacy_result = static_cast<uint32_t>(x0);
  if (legacy_result == kLegacyPrepareRejectedResult ||
      legacy_result == kLegacyPrepareRejectedResultSwapped) {
    const bool already_forced = ios_force_universal_prepare_command.exchange(
        true, std::memory_order_acq_rel);
    if (!already_forced) {
      XELOGW(
          "iOS JIT legacy prepare rejected (x0=0x{:08X}); switching to "
          "universal brk #0xf00d (x16=1)",
          legacy_result);
    }
    InvokeUniversalPrepareBreakpoint(aligned_start, aligned_length);
    MaybeRequestExternalJitDetach();
  }

  return true;
#else
  return false;
#endif
}

bool MaybeRequestExternalJitPrepare(void* address, size_t length) {
  bool expected = false;
  if (!ios_external_prepare_issued.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return true;
  }
  return RequestExternalJitPrepare(address, length);
}

bool AccessSatisfies(xe::memory::PageAccess actual,
                     xe::memory::PageAccess desired) {
  const uint32_t actual_bits = static_cast<uint32_t>(actual);
  const uint32_t desired_bits = static_cast<uint32_t>(desired);
  return (actual_bits & desired_bits) == desired_bits;
}

bool SetPageAlignedAccessWithExternalPrepareFallback(
    void* address, size_t length, xe::memory::PageAccess desired_access,
    const char* transition_name) {
  const bool use_txm_broker_path = IOSUseTXMBrokerPath();

  // Non-broker path must stay strict W^X and never request RWX max-protection
  // widening retries.
  if (use_txm_broker_path) {
    if (SetPageAlignedAccessWithMaxProtRetry(address, length, desired_access)) {
      return true;
    }
  } else {
    if (SetPageAlignedAccess(address, length, desired_access)) {
      return true;
    }
  }

  if (!use_txm_broker_path) {
    const int ios_major_version = IOSProductMajorVersion();
    if (ios_major_version > 0) {
      XELOGW("iOS JIT mprotect flip: {} denied on iOS {} without RWX retry or "
             "external brk fallback (non-broker path)",
             transition_name, ios_major_version);
    } else {
      XELOGW("iOS JIT mprotect flip: {} denied without RWX retry or external "
             "brk fallback (non-broker path)",
             transition_name);
    }
    return false;
  }

  if (!cvars::ios_jit_brk_prepare_fallback) {
    XELOGW("iOS JIT mprotect flip: {} denied with external brk fallback "
           "disabled on TXM path",
           transition_name);
    return false;
  }

  XELOGW("iOS JIT mprotect flip: {} denied, requesting external prepare via {}",
         transition_name, ExternalPrepareBreakpointDescription());
  if (!MaybeRequestExternalJitPrepare(address, length)) {
    return false;
  }

  // External JIT helpers (for example StikDebug scripts handling brk #0x69)
  // may only widen max protections. Retry setting current protection.
  if (SetPageAlignedAccessWithMaxProtRetry(address, length, desired_access)) {
    return true;
  }

  // If direct transition is still denied, only proceed if QueryProtect reports
  // the mapping already has at least the requested access.
  uintptr_t aligned_start = 0;
  size_t aligned_length = 0;
  if (!GetPageAlignedRange(address, length, aligned_start, aligned_length) ||
      !aligned_length) {
    return true;
  }

  size_t query_length = 0;
  xe::memory::PageAccess query_access = xe::memory::PageAccess::kNoAccess;
  const bool query_ok = xe::memory::QueryProtect(
      reinterpret_cast<void*>(aligned_start), query_length, query_access);
  if (query_ok && AccessSatisfies(query_access, desired_access)) {
    return true;
  }

  XELOGE(
      "iOS JIT mprotect flip: external prepare did not yield {} mapping "
      "addr=0x{:X} len=0x{:X} query_ok={} query_access={} query_len=0x{:X}",
      transition_name, aligned_start, static_cast<uint32_t>(aligned_length),
      query_ok, static_cast<uint32_t>(query_access),
      static_cast<uint32_t>(query_length));
  return false;
}

bool RegionLockRead(void* address, size_t length) {
  if (IOSUseTXMBrokerPath()) {
    return SetPageAlignedAccessWithMaxProtRetry(
        address, length, xe::memory::PageAccess::kReadOnly);
  }
  return SetPageAlignedAccess(address, length, xe::memory::PageAccess::kReadOnly);
}

bool RegionUnlockWrite(void* address, size_t length) {
  return SetPageAlignedAccessWithExternalPrepareFallback(
      address, length, xe::memory::PageAccess::kReadWrite, "RW transition");
}

bool RegionSetExec(void* address, size_t length) {
    
    
  return SetPageAlignedAccessWithExternalPrepareFallback(
      address, length, xe::memory::PageAccess::kExecuteReadOnly,
      "RX transition");
}
#endif  // XE_PLATFORM_IOS && XE_ARCH_ARM64

}  // namespace

// Define static constants for linking
const size_t A64CodeCache::kIndirectionTableSize;
#if XE_A64_INDIRECTION_64BIT
// On ARM64 platforms, this will be set dynamically during initialization
uintptr_t A64CodeCache::kIndirectionTableBase = 0x80000000;
#else
const uintptr_t A64CodeCache::kIndirectionTableBase;
#endif

A64CodeCache::A64CodeCache() = default;

A64CodeCache::~A64CodeCache() {
  if (indirection_table_base_) {
    xe::memory::DeallocFixed(indirection_table_base_, kIndirectionTableSize,
                             xe::memory::DeallocationType::kRelease);
  }

  // Unmap all views and close mapping.
  if (mapping_ != xe::memory::kFileMappingHandleInvalid) {
#if XE_PLATFORM_APPLE && XE_ARCH_ARM64
    // Apple ARM64 can use either:
    // 1) single MAP_JIT allocation (execute == write), or
    // 2) dual mapping (execute != write), including vm_remap fallback on iOS.
    if (generated_code_write_base_ &&
        generated_code_write_base_ != generated_code_execute_base_) {
      xe::memory::UnmapFileView(mapping_, generated_code_write_base_,
                                kGeneratedCodeSize);
      if (generated_code_execute_base_) {
        xe::memory::UnmapFileView(mapping_, generated_code_execute_base_,
                                  kGeneratedCodeSize);
      }
    } else if (generated_code_execute_base_) {
      xe::memory::DeallocFixed(generated_code_execute_base_, kGeneratedCodeSize,
                               xe::memory::DeallocationType::kRelease);
    }
#else
    // Other platforms use MapFileView/UnmapFileView
    if (generated_code_write_base_ &&
        generated_code_write_base_ != generated_code_execute_base_) {
      xe::memory::UnmapFileView(mapping_, generated_code_write_base_,
                                kGeneratedCodeSize);
    }
    if (generated_code_execute_base_) {
      xe::memory::UnmapFileView(mapping_, generated_code_execute_base_,
                                kGeneratedCodeSize);
    }
#endif
    xe::memory::CloseFileMappingHandle(mapping_, file_name_);
    mapping_ = xe::memory::kFileMappingHandleInvalid;
  }
}

bool A64CodeCache::Initialize() {
  generated_code_uses_vm_remap_fallback_ = false;
  generated_code_uses_mprotect_flip_ = false;

#if XE_A64_INDIRECTION_64BIT
  // On ARM64 platforms, allocate the indirection table wherever the OS allows,
  // then update our base address to match. Reserve as no-access and commit
  // executable ranges lazily to reduce upfront VM pressure on iOS.

  indirection_table_base_ = reinterpret_cast<uint8_t*>(xe::memory::AllocFixed(
      nullptr, kIndirectionTableSize, xe::memory::AllocationType::kReserve,
      xe::memory::PageAccess::kNoAccess));

  if (!indirection_table_base_) {
    XELOGE("Unable to reserve indirection table at any address (size=0x{:X})",
           static_cast<uint32_t>(kIndirectionTableSize));
    return false;
  }

  // Keep kIndirectionTableBase as 0x80000000 for offset calculations
  // Store the actual allocated address separately
  indirection_table_actual_base_ =
      reinterpret_cast<uintptr_t>(indirection_table_base_);
#if XE_A64_INDIRECTION_64BIT
  indirection_table_base_bias_ =
      indirection_table_actual_base_ -
      (static_cast<uintptr_t>(kIndirectionTableBase) * 2);
#endif
#else
  // Other platforms: try to allocate at the preferred address first.
  indirection_table_base_ = reinterpret_cast<uint8_t*>(xe::memory::AllocFixed(
      reinterpret_cast<void*>(kIndirectionTableBase), kIndirectionTableSize,
      xe::memory::AllocationType::kReserve,
      xe::memory::PageAccess::kReadWrite));
  if (!indirection_table_base_) {
    XELOGW("Preferred indirection table base unavailable; falling back");
    indirection_table_base_ = reinterpret_cast<uint8_t*>(xe::memory::AllocFixed(
        nullptr, kIndirectionTableSize, xe::memory::AllocationType::kReserve,
        xe::memory::PageAccess::kReadWrite));
  }
  if (!indirection_table_base_) {
    XELOGE("Unable to allocate code cache indirection table");
    XELOGE("Tried preferred range {:X}-{:X} with fallback to OS-chosen",
           static_cast<uint64_t>(kIndirectionTableBase),
           kIndirectionTableBase + kIndirectionTableSize);
    return false;
  }
  indirection_table_actual_base_ =
      reinterpret_cast<uintptr_t>(indirection_table_base_);
#if XE_A64_INDIRECTION_64BIT
  indirection_table_base_bias_ =
      indirection_table_actual_base_ -
      (static_cast<uintptr_t>(kIndirectionTableBase) * 2);
#endif
#endif

  if (ShouldLogIndirectionTable()) {
    XELOGI(
        "A64 indirection table: guest_base=0x{:08X} table_base=0x{:016X} "
        "size=0x{:X} entry_bytes={}",
        static_cast<uint32_t>(kIndirectionTableBase),
        static_cast<uint64_t>(indirection_table_actual_base_),
        static_cast<uint32_t>(kIndirectionTableSize),
        static_cast<uint32_t>(kIndirectionEntrySize));
  }

  // Create mmap file. This allows us to share the code cache with the debugger.
  file_name_ = fmt::format("xenia_code_cache_{}", Clock::QueryHostTickCount());
  mapping_ = xe::memory::CreateFileMappingHandle(
      file_name_, kGeneratedCodeSize, xe::memory::PageAccess::kExecuteReadWrite,
      false);
  if (mapping_ == xe::memory::kFileMappingHandleInvalid) {
    XELOGE("Unable to create code cache mmap");
    return false;
  }

  // Map generated code region into the file. Pages are committed as required.
  if (xe::memory::IsWritableExecutableMemoryPreferred()) {
#if XE_PLATFORM_APPLE && XE_ARCH_ARM64
    // On macOS ARM64, always use OS-chosen MAP_JIT memory.
    generated_code_execute_base_ = reinterpret_cast<uint8_t*>(
        xe::memory::AllocFixed(nullptr, kGeneratedCodeSize,
                               xe::memory::AllocationType::kReserveCommit,
                               xe::memory::PageAccess::kExecuteReadWrite));
    generated_code_write_base_ = generated_code_execute_base_;
    if (!generated_code_execute_base_ || !generated_code_write_base_) {
      XELOGE("Unable to allocate code cache generated code storage");
      return false;
    }
    // On macOS ARM64, verify the memory is properly allocated for MAP_JIT
#else
    generated_code_execute_base_ =
        reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
            mapping_, reinterpret_cast<void*>(kGeneratedCodeExecuteBase),
            kGeneratedCodeSize, xe::memory::PageAccess::kExecuteReadWrite, 0));
    if (!generated_code_execute_base_) {
      XELOGW(
          "Fixed address mapping for generated code failed, trying OS-chosen "
          "address");
      generated_code_execute_base_ =
          reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
              mapping_, nullptr, kGeneratedCodeSize,
              xe::memory::PageAccess::kExecuteReadWrite, 0));
    }
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
#endif
  } else {
#if XE_PLATFORM_APPLE && XE_ARCH_ARM64
#if XE_PLATFORM_IOS
    // iOS JIT path:
    // - TXM on iOS 26+: dual-map with optional external BRK prepare.
    // - Everything else: strict W^X via single-view mprotect flips, without
    //   relying on external BRK/broker callbacks.
    const bool has_txm = IOSHasTXM();
    const int ios_major_version = IOSProductMajorVersion();
    const bool use_txm_broker_path = IOSUseTXMBrokerPath();
    // On iOS below 26, dual-mapping leaves a writable alias around that can
    // violate W^X enforcement on TXM devices (leading to a fault on first JIT
    // entry). Prefer the deterministic mprotect-flip path instead.
    const bool should_try_dual_map = use_txm_broker_path;
    const bool force_mprotect_flip_requested =
        cvars::ios_jit_non_txm_force_mprotect_flip;
    const bool force_mprotect_flip =
        force_mprotect_flip_requested && !has_txm;

    if (use_txm_broker_path) {
      if (ios_major_version > 0) {
        XELOGI("iOS JIT: TXM detected on iOS {}; using broker-capable path",
               ios_major_version);
      } else {
        XELOGI("iOS JIT: TXM detected; using broker-capable path");
      }
    } else if (has_txm) {
      if (ios_major_version > 0) {
        XELOGI(
            "iOS JIT: TXM detected on iOS {}; using non-broker W^X defaults",
            ios_major_version);
      } else {
        XELOGI("iOS JIT: TXM detected with unknown OS version; using "
               "non-broker W^X defaults");
      }
    } else {
      if (ios_major_version > 0) {
        XELOGI(
            "iOS JIT: non-TXM path on iOS {}; using non-broker W^X defaults",
            ios_major_version);
      } else {
        XELOGI("iOS JIT: non-TXM path; using non-broker W^X defaults");
      }
    }

    if (force_mprotect_flip_requested && has_txm) {
      XELOGW(
          "iOS JIT: ignoring non-TXM mprotect-flip override because TXM is "
          "active");
    } else if (force_mprotect_flip) {
      XELOGI("iOS JIT: forcing non-TXM mprotect-flip path by config");
    } else if (!should_try_dual_map) {
      XELOGI("iOS JIT: using mprotect-flip path (non-broker W^X)");
    }

    if (!force_mprotect_flip && should_try_dual_map) {
      generated_code_execute_base_ = reinterpret_cast<uint8_t*>(
          mmap(nullptr, kGeneratedCodeSize, PROT_READ | PROT_EXEC,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      if (generated_code_execute_base_ != MAP_FAILED) {
        XELOGI("iOS JIT dual-map setup (RX base): {:p}",
               static_cast<void*>(generated_code_execute_base_));

        if (use_txm_broker_path && cvars::ios_jit_brk_prepare_fallback) {
          size_t prepare_size = kGeneratedCodeSize;
          if (cvars::ios_jit_initial_external_prepare_bytes > 0) {
            prepare_size = static_cast<size_t>(
                cvars::ios_jit_initial_external_prepare_bytes);
            if (prepare_size > kGeneratedCodeSize) {
              prepare_size = kGeneratedCodeSize;
            }
          }
          XELOGI(
              "iOS JIT TXM: requesting one-shot external prepare via {} "
              "for RX execute cache window (0x{:X} bytes)",
              ExternalPrepareBreakpointDescription(),
              static_cast<uint32_t>(prepare_size));
          if (!MaybeRequestExternalJitPrepare(generated_code_execute_base_,
                                              prepare_size)) {
            XELOGW("iOS JIT TXM: initial external prepare request failed");
          }
        } else if (use_txm_broker_path) {
          XELOGI("iOS JIT TXM: external BRK prepare fallback disabled by "
                 "config");
        }

        vm_address_t remap_addr = 0;
        vm_prot_t cur_prot = 0;
        vm_prot_t max_prot = 0;
        const kern_return_t remap_kr =
            vm_remap(mach_task_self(), &remap_addr, kGeneratedCodeSize,
                     0,  // mask
                     VM_FLAGS_ANYWHERE, mach_task_self(),
                     reinterpret_cast<vm_address_t>(generated_code_execute_base_),
                     FALSE,  // copy = false (share the same physical pages)
                     &cur_prot, &max_prot, VM_INHERIT_NONE);
        if (remap_kr == KERN_SUCCESS) {
          auto* remap_write_base = reinterpret_cast<uint8_t*>(remap_addr);
          if (mprotect(remap_write_base, kGeneratedCodeSize,
                       PROT_READ | PROT_WRITE) == 0) {
            generated_code_write_base_ = remap_write_base;
            generated_code_uses_vm_remap_fallback_ = true;
            XELOGI("iOS JIT dual-mapping active: execute={:p} write={:p}",
                   static_cast<void*>(generated_code_execute_base_),
                   static_cast<void*>(generated_code_write_base_));
            XELOGI(
                "A64 code cache: iOS dual mapping active; skipping "
                "incremental commit/protect updates");
          } else {
            XELOGE(
                "iOS JIT dual-mapping: mprotect RW alias failed "
                "addr=0x{:X} len=0x{:X} err={} ({})",
                reinterpret_cast<uintptr_t>(remap_write_base),
                static_cast<uint32_t>(kGeneratedCodeSize), errno,
                std::strerror(errno));
            vm_deallocate(mach_task_self(), remap_addr, kGeneratedCodeSize);
          }
        } else {
          XELOGW("iOS JIT dual-mapping: vm_remap failed (kr={})", remap_kr);
        }
      } else {
        generated_code_execute_base_ = nullptr;
        XELOGW("iOS JIT dual-map setup (RX base) failed");
      }
    } else {
      XELOGI("iOS JIT: dual-map disabled by mprotect-flip override");
    }

    if (!generated_code_execute_base_ || !generated_code_write_base_) {
      if (generated_code_execute_base_) {
        munmap(generated_code_execute_base_, kGeneratedCodeSize);
        generated_code_execute_base_ = nullptr;
      }
      generated_code_write_base_ = nullptr;
      generated_code_uses_vm_remap_fallback_ = false;

      // Dual-map setup failed (or was disabled). Fall back to single-view
      // mprotect flips using strict W^X (RW while writing, RX when published).
      ios_external_prepare_issued.store(false, std::memory_order_release);
      ios_external_detach_issued.store(false, std::memory_order_release);
      // For W^X flips, prefer starting from an RX mapping (so EXECUTE is
      // present in the mapping's max protections on iOS), then temporarily
      // switching pages to RW for writes.
      generated_code_write_base_ = reinterpret_cast<uint8_t*>(
          mmap(nullptr, kGeneratedCodeSize, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      if (generated_code_write_base_ == MAP_FAILED) {
          generated_code_write_base_ = nullptr;
        XELOGE("Unable to allocate iOS JIT code cache (RX mapping)");
        return false;
      }
      generated_code_execute_base_ = generated_code_write_base_;
      generated_code_uses_mprotect_flip_ = true;
      if (use_txm_broker_path) {
        XELOGI("iOS JIT mprotect-flip fallback active (TXM/broker path)");
      } else if (has_txm) {
        XELOGI(
            "iOS JIT mprotect-flip fallback active (TXM, non-broker path)");
      } else {
        XELOGI(
            "iOS JIT mprotect-flip fallback active (non-TXM, no external BRK)");
      }
      XELOGI("iOS JIT mprotect-flip mapping (RX base): {:p}",
             static_cast<void*>(generated_code_execute_base_));
    }
#else
    // On Apple ARM64, use OS-chosen addresses for the dual-mapping views.
    // Try anonymous RWX first. On iOS this may succeed when debug/JIT state
    // is enabled (for instance via debugger attachment).
    generated_code_execute_base_ = reinterpret_cast<uint8_t*>(
        mmap(nullptr, kGeneratedCodeSize, PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (generated_code_execute_base_ != MAP_FAILED) {
      generated_code_write_base_ = generated_code_execute_base_;
      XELOGI("anonymous RWX JIT mapping: base={:p}",
             static_cast<void*>(generated_code_execute_base_));
    } else {
      generated_code_execute_base_ = nullptr;
    }

    // If anonymous RWX is unavailable, try dual mapping.
    // Try the shm_open-based MapFileView first (works on macOS and iOS with
    // CS_DEBUGGED).  If that fails (e.g. iOS sandbox restrictions on
    // shm_open), fall back to vm_remap, which is the technique used by
    // UTM/QEMU and DolphiniOS for JIT on iOS.
    if (!generated_code_execute_base_ || !generated_code_write_base_) {
      generated_code_execute_base_ = reinterpret_cast<uint8_t*>(
          xe::memory::MapFileView(mapping_, nullptr, kGeneratedCodeSize,
                                  xe::memory::PageAccess::kExecuteReadOnly, 0));
      generated_code_write_base_ = reinterpret_cast<uint8_t*>(
          xe::memory::MapFileView(mapping_, nullptr, kGeneratedCodeSize,
                                  xe::memory::PageAccess::kReadWrite, 0));
    }
    if (!generated_code_execute_base_ || !generated_code_write_base_) {
      // shm_open dual-mapping failed — try vm_remap fallback.
      // 1. Allocate an anonymous RW region.
      // 2. Use vm_remap to create a second mapping of the same pages.
      // 3. Set the second mapping to RX.
      XELOGW("shm_open dual-mapping failed, trying vm_remap fallback");
      if (generated_code_execute_base_) {
        xe::memory::UnmapFileView(mapping_, generated_code_execute_base_,
                                  kGeneratedCodeSize);
        generated_code_execute_base_ = nullptr;
      }
      if (generated_code_write_base_) {
        xe::memory::UnmapFileView(mapping_, generated_code_write_base_,
                                  kGeneratedCodeSize);
        generated_code_write_base_ = nullptr;
      }

      // Allocate anonymous RW region.
      generated_code_write_base_ = reinterpret_cast<uint8_t*>(
          mmap(nullptr, kGeneratedCodeSize, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      if (generated_code_write_base_ == MAP_FAILED) {
        generated_code_write_base_ = nullptr;
        XELOGE("vm_remap fallback: failed to allocate RW region");
        return false;
      }

      // Try to widen max protection on the write mapping so the remapped alias
      // can be switched to executable on iOS. Some kernels may refuse execute
      // in the max protection set for plain RW mappings.
      constexpr vm_prot_t kWriteProt = VM_PROT_READ | VM_PROT_WRITE;
      constexpr vm_prot_t kExecProt = VM_PROT_READ | VM_PROT_EXECUTE;
      constexpr vm_prot_t kWriteExecProt =
          VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
      kern_return_t kr_max =
          vm_protect(mach_task_self(),
                     reinterpret_cast<vm_address_t>(generated_code_write_base_),
                     kGeneratedCodeSize, TRUE, kWriteExecProt);
      if (kr_max == KERN_SUCCESS) {
        // Keep the write mapping non-executable for W^X.
        vm_protect(mach_task_self(),
                   reinterpret_cast<vm_address_t>(generated_code_write_base_),
                   kGeneratedCodeSize, FALSE, kWriteProt);
      } else {
        XELOGW(
            "vm_remap fallback: couldn't widen write mapping max prot to RWX "
            "(kr={})",
            kr_max);
      }

      // Create a second mapping of the same physical pages via vm_remap.
      vm_address_t remap_addr = 0;
      vm_prot_t cur_prot, max_prot;
      kern_return_t kr =
          vm_remap(mach_task_self(), &remap_addr, kGeneratedCodeSize,
                   0,  // mask
                   VM_FLAGS_ANYWHERE, mach_task_self(),
                   reinterpret_cast<vm_address_t>(generated_code_write_base_),
                   FALSE,  // copy = false (share the same physical pages)
                   &cur_prot, &max_prot, VM_INHERIT_NONE);
      if (kr != KERN_SUCCESS) {
        XELOGE("vm_remap fallback: vm_remap failed with error {}", kr);
        munmap(generated_code_write_base_, kGeneratedCodeSize);
        generated_code_write_base_ = nullptr;
        return false;
      }

      // Ensure execute permission is in the max set for the remapped alias
      // before setting current protection to RX.
      kern_return_t kr_exec_max = vm_protect(
          mach_task_self(), remap_addr, kGeneratedCodeSize, TRUE, kExecProt);
      if (kr_exec_max != KERN_SUCCESS) {
        XELOGW("vm_remap fallback: vm_protect set-max RX failed (kr={})",
               kr_exec_max);
      }

      // Set the remapped region to read-execute.
      kern_return_t kr_exec = vm_protect(mach_task_self(), remap_addr,
                                         kGeneratedCodeSize, FALSE, kExecProt);
      if (kr_exec != KERN_SUCCESS) {
        XELOGE("vm_remap fallback: vm_protect RX failed (kr={})", kr_exec);
        vm_deallocate(mach_task_self(), remap_addr, kGeneratedCodeSize);
        munmap(generated_code_write_base_, kGeneratedCodeSize);
        generated_code_write_base_ = nullptr;
        return false;
      }

      // Verify protections as seen by QueryProtect. If this still isn't
      // executable, fail fast rather than crashing on first thunk call.
      size_t exec_len = 0;
      xe::memory::PageAccess exec_access = xe::memory::PageAccess::kNoAccess;
      const bool exec_query_ok = xe::memory::QueryProtect(
          reinterpret_cast<void*>(remap_addr), exec_len, exec_access);
      const bool exec_access_ok =
          exec_access == xe::memory::PageAccess::kExecuteReadOnly ||
          exec_access == xe::memory::PageAccess::kExecuteReadWrite;
      if (!exec_query_ok || !exec_access_ok) {
        XELOGE(
            "vm_remap fallback: remapped alias still not executable "
            "(query_ok={} access={} size=0x{:X} cur=0x{:X} max=0x{:X})",
            exec_query_ok, static_cast<uint32_t>(exec_access),
            static_cast<uint32_t>(exec_len), static_cast<uint32_t>(cur_prot),
            static_cast<uint32_t>(max_prot));
        vm_deallocate(mach_task_self(), remap_addr, kGeneratedCodeSize);
        munmap(generated_code_write_base_, kGeneratedCodeSize);
        generated_code_write_base_ = nullptr;
        return false;
      }

      generated_code_execute_base_ = reinterpret_cast<uint8_t*>(remap_addr);
      generated_code_uses_vm_remap_fallback_ = true;
      XELOGI("vm_remap JIT dual-mapping: write={:p} execute={:p}",
             static_cast<void*>(generated_code_write_base_),
             static_cast<void*>(generated_code_execute_base_));
      XELOGI(
          "A64 code cache: vm_remap fallback active; skipping incremental "
          "commit/protect updates");
    }
#endif  // XE_PLATFORM_IOS
#else
    generated_code_execute_base_ =
        reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
            mapping_, reinterpret_cast<void*>(kGeneratedCodeExecuteBase),
            kGeneratedCodeSize, xe::memory::PageAccess::kExecuteReadOnly, 0));
    if (!generated_code_execute_base_) {
      XELOGW(
          "Fixed address mapping for generated code failed, trying OS-chosen "
          "address");
      generated_code_execute_base_ = reinterpret_cast<uint8_t*>(
          xe::memory::MapFileView(mapping_, nullptr, kGeneratedCodeSize,
                                  xe::memory::PageAccess::kExecuteReadOnly, 0));
    }
    generated_code_write_base_ =
        reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
            mapping_, reinterpret_cast<void*>(kGeneratedCodeWriteBase),
            kGeneratedCodeSize, xe::memory::PageAccess::kReadWrite, 0));
    if (!generated_code_write_base_) {
      XELOGW(
          "Fixed address mapping for generated code failed, trying OS-chosen "
          "address");
      generated_code_write_base_ = reinterpret_cast<uint8_t*>(
          xe::memory::MapFileView(mapping_, nullptr, kGeneratedCodeSize,
                                  xe::memory::PageAccess::kReadWrite, 0));
    }
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
#endif
  }

  // Preallocate the function map to a large, reasonable size.
  generated_code_map_.reserve(kMaximumFunctionCount);

  return true;
}

void A64CodeCache::set_indirection_default(uint32_t default_value) {
#if XE_A64_INDIRECTION_64BIT
  // On ARM64 platforms, we extend 32-bit values to 64-bit
  indirection_default_value_ = default_value;
#else
  indirection_default_value_ = default_value;
#endif
}

#if XE_A64_INDIRECTION_64BIT
void A64CodeCache::set_indirection_default_64(uint64_t default_value) {
  indirection_default_value_ = default_value;
}
#endif

void A64CodeCache::AddIndirection(uint32_t guest_address,
                                  uint32_t host_address) {
#if XE_A64_INDIRECTION_64BIT
  // On ARM64 platforms, delegate to the 64-bit version
  AddIndirection64(guest_address, host_address);
#else
  if (!indirection_table_base_) {
    return;
  }

  uint32_t* indirection_slot = reinterpret_cast<uint32_t*>(
      indirection_table_base_ + (guest_address - kIndirectionTableBase));
  *indirection_slot = host_address;
#endif
}

#if XE_A64_INDIRECTION_64BIT
void A64CodeCache::AddIndirection64(uint32_t guest_address,
                                    uint64_t host_address) {
  if (!indirection_table_base_) {
    return;
  }

  if (guest_address < kIndirectionTableBase) {
    XELOGE(
        "A64CodeCache::AddIndirection64: guest_address 0x{:08X} below base "
        "0x{:08X}",
        guest_address, static_cast<uint32_t>(kIndirectionTableBase));
    return;
  }

  const uint64_t guest_delta = guest_address - kIndirectionTableBase;
  if (guest_delta & 0x3) {
    XELOGW(
        "A64CodeCache::AddIndirection64: guest_address 0x{:08X} not 4-byte "
        "aligned (delta=0x{:X})",
        guest_address, guest_delta);
  }

  // Calculate offset from the logical base (0x80000000), not from actual table
  // address.
  const uint64_t guest_offset = (guest_delta >> 2) * kIndirectionEntrySize;
  if (guest_offset + kIndirectionEntrySize > kIndirectionTableSize) {
    XELOGE(
        "A64CodeCache::AddIndirection64: guest_address 0x{:08X} offset 0x{:X} "
        "exceeds table size 0x{:X}",
        guest_address, guest_offset,
        static_cast<uint32_t>(kIndirectionTableSize));
    return;
  }

  uint64_t* indirection_slot =
      reinterpret_cast<uint64_t*>(indirection_table_base_ + guest_offset);
  *indirection_slot = host_address;

  if (ShouldLogIndirectionTable()) {
    XELOGI(
        "A64 indirection add: guest=0x{:08X} delta=0x{:X} offset=0x{:X} "
        "slot=0x{:016X} host=0x{:016X}",
        guest_address, guest_delta, guest_offset,
        reinterpret_cast<uint64_t>(indirection_slot), host_address);
  }
}
#endif

void A64CodeCache::CommitExecutableRange(uint32_t guest_low,
                                         uint32_t guest_high) {
  if (!indirection_table_base_) {
    XELOGE("CommitExecutableRange: indirection_table_base_ is null!");
    return;
  }

#if XE_A64_INDIRECTION_64BIT
  // On ARM64 platforms: use offset-based addressing from guest base
  // (0x80000000)
  static const uintptr_t kGuestAddressBase = 0x80000000;

  // Calculate offsets from the guest address base, not the table base
  if (guest_low < kGuestAddressBase) {
    XELOGE(
        "CommitExecutableRange: guest_low 0x{:08X} is below guest base "
        "0x{:08X}",
        guest_low, kGuestAddressBase);
    return;
  }

  uint32_t start_offset =
      ((guest_low - kGuestAddressBase) >> 2) * kIndirectionEntrySize;
  uint32_t size = ((guest_high - guest_low) >> 2) * kIndirectionEntrySize;

  // Sanity check bounds; the table should fully cover the XEX guest range now.
  if (start_offset + size > kIndirectionTableSize) {
    XELOGE(
        "CommitExecutableRange: range [0x{:08X}, 0x{:08X}) exceeds table (size "
        "0x{:X})",
        guest_low, guest_high, static_cast<uint32_t>(kIndirectionTableSize));
    return;
  }

  // The memory should already be allocated, just fill with default value
  void* target_memory = indirection_table_base_ + start_offset;
  if (!xe::memory::AllocFixed(target_memory, size,
                              xe::memory::AllocationType::kCommit,
                              xe::memory::PageAccess::kReadWrite)) {
    XELOGE(
        "CommitExecutableRange: failed to commit indirection table pages "
        "offset=0x{:X} size=0x{:X}",
        start_offset, size);
    return;
  }
  uint64_t* p = reinterpret_cast<uint64_t*>(target_memory);
  uint32_t entry_count = size / kIndirectionEntrySize;
  for (uint32_t i = 0; i < entry_count; i++) {
    p[i] = indirection_default_value_;
  }
#else
  // Other platforms: use 32-bit entries
  uint32_t start_offset = (guest_low - kIndirectionTableBase);
  uint32_t size = (guest_high - guest_low);

  xe::memory::AllocFixed(indirection_table_base_ + start_offset, size,
                         xe::memory::AllocationType::kCommit,
                         xe::memory::PageAccess::kReadWrite);

  uint32_t* p = reinterpret_cast<uint32_t*>(indirection_table_base_);
  for (uint32_t address = guest_low; address < guest_high; address += 4) {
    p[(address - kIndirectionTableBase) / 4] = indirection_default_value_;
  }
#endif
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
  // Hold a lock while we bump the pointers up. This is important as the
  // unwind table requires entries AND code to be sorted in order.
  [[maybe_unused]] size_t low_mark;
  size_t high_mark;
  size_t write_span_length = 0;
  uint8_t* code_execute_address;
  UnwindReservation unwind_reservation;
  {
    auto global_lock = global_critical_region_.Acquire();

    low_mark = generated_code_offset_;

    // In single-view mprotect-flip mode on iOS, changing page protections to RW
    // makes the affected pages non-executable. If we ever flip a page that
    // already contains published JIT code, other guest threads may fault
    // fetching instructions from it. Avoid this by ensuring each placed range
    // starts on a fresh page and is never written to again after being
    // published RX.
#if XE_PLATFORM_IOS && XE_ARCH_ARM64
    if (generated_code_uses_mprotect_flip_) {
      generated_code_offset_ =
          xe::align(generated_code_offset_, xe::memory::page_size());
    }
#endif

    // Reserve code.
    // Always move the code to land on 16b alignment.
    code_execute_address =
        generated_code_execute_base_ + generated_code_offset_;
    code_execute_address_out = code_execute_address;
    uint8_t* code_write_address =
        generated_code_write_base_ + generated_code_offset_;
    code_write_address_out = code_write_address;
    generated_code_offset_ += xe::round_up(func_info.code_size.total, 16);

    auto tail_write_address =
        generated_code_write_base_ + generated_code_offset_;

    // Reserve unwind info.
    // We go on the high size of the unwind info as we don't know how big we
    // need it, and a few extra bytes of padding isn't the worst thing.
    unwind_reservation = RequestUnwindReservation(generated_code_write_base_ +
                                                  generated_code_offset_);
    generated_code_offset_ += xe::round_up(unwind_reservation.data_size, 16);

    auto end_write_address =
        generated_code_write_base_ + generated_code_offset_;
    write_span_length =
        static_cast<size_t>(end_write_address - code_write_address);

    high_mark = generated_code_offset_;

    // Store in map. It is maintained in sorted order of host PC dependent on
    // us also being append-only.
    generated_code_map_.emplace_back(
        (uint64_t(code_execute_address - generated_code_execute_base_) << 32) |
            generated_code_offset_,
        function_info);

    // TODO(DrChat): The following code doesn't really need to be under the
    // global lock except for PlaceCode (but it depends on the previous code
    // already being ran)

    // If we are going above the high water mark of committed memory, commit
    // some more. It's ok if multiple threads do this, as redundant commits
    // aren't harmful.
    size_t old_commit_mark, new_commit_mark;
    do {
      old_commit_mark = generated_code_commit_mark_;
      if (high_mark <= old_commit_mark) break;

      new_commit_mark = old_commit_mark + 16_MiB;
      if (!generated_code_uses_vm_remap_fallback_ &&
          !generated_code_uses_mprotect_flip_) {
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
      }
    } while (generated_code_commit_mark_.compare_exchange_weak(
        old_commit_mark, new_commit_mark));

#if XE_PLATFORM_IOS && XE_ARCH_ARM64
    if (generated_code_uses_mprotect_flip_) {
      if (!RegionLockRead(code_write_address, write_span_length)) {
        XELOGE("iOS JIT mprotect flip: failed to lock code range for writes");
        assert_always();
      }
      if (!RegionUnlockWrite(code_write_address, write_span_length)) {
        XELOGE("iOS JIT mprotect flip: failed to enable writes before code "
               "publish");
        assert_always();
      }
    }
#endif

    // Copy code and fill padding while in write mode on MAP_JIT.
    // pthread_jit_write_protect_np is only available on macOS ARM64, not iOS.
    // On iOS the dual-mapping (split W^X) path is used instead.
#if XE_PLATFORM_APPLE && !XE_PLATFORM_IOS && defined(__aarch64__)
    const bool jit_write =
        (generated_code_execute_base_ == generated_code_write_base_);
    if (jit_write) {
      pthread_jit_write_protect_np(0);
    }
#endif
    CopyMachineCode(code_write_address, machine_code,
                    func_info.code_size.total);
    if (end_write_address > tail_write_address) {
      std::memset(tail_write_address, 0x00,
                  static_cast<size_t>(end_write_address - tail_write_address));
    }
#if XE_PLATFORM_APPLE && !XE_PLATFORM_IOS && defined(__aarch64__)
    if (jit_write) {
      pthread_jit_write_protect_np(1);
    }
#endif

    // Notify subclasses of placed code.
    PlaceCode(guest_address, machine_code, func_info, code_execute_address,
              unwind_reservation);

#if XE_PLATFORM_IOS && XE_ARCH_ARM64
    if (generated_code_uses_mprotect_flip_) {
      // Transition the write-enabled mapping back to RX. Keeping an
      // intermediate non-executable state can fault concurrent execution on
      // iOS 18 and below.
      if (!RegionSetExec(code_execute_address, write_span_length)) {
        XELOGE("iOS JIT mprotect flip: failed to restore RX after code write");
        assert_always();
      }
    }
#endif
  }

  // Now that everything is ready, fix up the indirection table.
  // Note that we do support code that doesn't have an indirection fixup, so
  // ignore those when we see them.
  if (guest_address && indirection_table_base_) {
#if XE_A64_INDIRECTION_64BIT
    // On ARM64 platforms, map guest addresses to table offsets using logical
    // base kIndirectionTableBase remains 0x80000000 for calculation purposes

    // Calculate offset from the logical guest base (0x80000000)
    if (guest_address < kIndirectionTableBase) {
      return;
    }

    uintptr_t guest_diff = guest_address - kIndirectionTableBase;
    uintptr_t guest_offset =
        (guest_diff >> 2) * kIndirectionEntrySize;  // 8-byte entries
    uintptr_t slot_address =
        reinterpret_cast<uintptr_t>(indirection_table_base_) + guest_offset;
    uint64_t* indirection_slot = reinterpret_cast<uint64_t*>(slot_address);

    // Check if the slot address is within bounds
    uintptr_t table_end = reinterpret_cast<uintptr_t>(indirection_table_base_) +
                          kIndirectionTableSize;
    if (slot_address >= table_end) {
      return;
    }

    *indirection_slot = reinterpret_cast<uint64_t>(code_execute_address);
#else
    uint32_t* indirection_slot = reinterpret_cast<uint32_t*>(
        indirection_table_base_ + (guest_address - kIndirectionTableBase));
    *indirection_slot =
        uint32_t(reinterpret_cast<uint64_t>(code_execute_address));
#endif
  }
}

uint32_t A64CodeCache::PlaceData(const void* data, size_t length) {
  // Hold a lock while we bump the pointers up.
  size_t high_mark;
  size_t reserved_length = 0;
  uint8_t* data_address = nullptr;
  {
    auto global_lock = global_critical_region_.Acquire();

    // See PlaceGuestCode: in iOS mprotect-flip mode, never write into already
    // published pages.
#if XE_PLATFORM_IOS && XE_ARCH_ARM64
    if (generated_code_uses_mprotect_flip_) {
      generated_code_offset_ =
          xe::align(generated_code_offset_, xe::memory::page_size());
    }
#endif

    // Reserve code.
    // Always move the code to land on 16b alignment.
    reserved_length = xe::round_up(length, 16);
    data_address = generated_code_write_base_ + generated_code_offset_;
    generated_code_offset_ += reserved_length;

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
    if (!generated_code_uses_vm_remap_fallback_ &&
        !generated_code_uses_mprotect_flip_) {
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
    }
  } while (generated_code_commit_mark_.compare_exchange_weak(old_commit_mark,
                                                             new_commit_mark));

#if XE_PLATFORM_IOS && XE_ARCH_ARM64
  if (generated_code_uses_mprotect_flip_) {
    if (!RegionLockRead(data_address, reserved_length)) {
      XELOGE("iOS JIT mprotect flip: failed to lock data range for writes");
      assert_always();
    }
    if (!RegionUnlockWrite(data_address, reserved_length)) {
      XELOGE("iOS JIT mprotect flip: failed to enable writes before data "
             "publish");
      assert_always();
    }
  }
#endif

  // Copy data.
  // pthread_jit_write_protect_np is only available on macOS ARM64, not iOS.
  // On iOS the dual-mapping (split W^X) path is used instead.
#if XE_PLATFORM_APPLE && !XE_PLATFORM_IOS && defined(__aarch64__)
  if (generated_code_execute_base_ == generated_code_write_base_) {
    pthread_jit_write_protect_np(0);
    std::memcpy(data_address, data, length);
    pthread_jit_write_protect_np(1);
  } else {
    std::memcpy(data_address, data, length);
  }
#else
  std::memcpy(data_address, data, length);
#endif

#if XE_PLATFORM_IOS && XE_ARCH_ARM64
  if (generated_code_uses_mprotect_flip_) {
    if (!RegionLockRead(data_address, reserved_length)) {
      XELOGE("iOS JIT mprotect flip: failed to lock data range before RX");
      assert_always();
    }
    if (!RegionSetExec(data_address, reserved_length)) {
      XELOGE("iOS JIT mprotect flip: failed to restore RX after data write");
      assert_always();
    }
  }
#endif

  return uint32_t(uintptr_t(data_address));
}

GuestFunction* A64CodeCache::LookupFunction(uint64_t host_pc) {
  if (generated_code_map_.empty()) {
    return nullptr;
  }
  const uint64_t code_base = execute_base_address();
  const uint64_t code_end = code_base + total_size();
  if (host_pc < code_base || host_pc >= code_end) {
    return nullptr;
  }
  uint32_t key = uint32_t(host_pc - code_base);
  void* fn_entry = std::bsearch(
      &key, generated_code_map_.data(), generated_code_map_.size(),
      sizeof(generated_code_map_[0]),
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
