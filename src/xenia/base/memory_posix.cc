/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/memory.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstddef>
#include <cerrno>

#include "xenia/base/math.h"
#include "xenia/base/platform.h"
#include "xenia/base/string.h"

#if XE_PLATFORM_MAC
#define MAP_ANONYMOUS MAP_ANON
#endif

#if XE_PLATFORM_ANDROID
#include <dlfcn.h>
#include <linux/ashmem.h>
#include <string.h>
#include <sys/ioctl.h>

#include "xenia/base/main_android.h"
#endif

namespace xe {
namespace memory {

#if XE_PLATFORM_ANDROID
// May be null if no dynamically loaded functions are required.
static void* libandroid_;
// API 26+.
static int (*android_ASharedMemory_create_)(const char* name, size_t size);
static int (*android_ASharedMemory_setProt_)(int fd, int prot);

void AndroidInitialize() {
  if (xe::GetAndroidApiLevel() >= 26) {
    libandroid_ = dlopen("libandroid.so", RTLD_NOW);
    assert_not_null(libandroid_);
    if (libandroid_) {
      android_ASharedMemory_create_ =
          reinterpret_cast<decltype(android_ASharedMemory_create_)>(
              dlsym(libandroid_, "ASharedMemory_create"));
      assert_not_null(android_ASharedMemory_create_);
      android_ASharedMemory_setProt_ =
          reinterpret_cast<decltype(android_ASharedMemory_setProt_)>(
              dlsym(libandroid_, "ASharedMemory_setProt"));
    }
  }
}

void AndroidShutdown() {
  android_ASharedMemory_create_ = nullptr;
  android_ASharedMemory_setProt_ = nullptr;
  if (libandroid_) {
    dlclose(libandroid_);
    libandroid_ = nullptr;
  }
}
#endif

size_t page_size() {
  const long size = sysconf(_SC_PAGESIZE);
  return size > 0 ? static_cast<size_t>(size) : size_t(4096);
}
size_t allocation_granularity() { return page_size(); }

uint32_t ToPosixProtectFlags(PageAccess access) {
  switch (access) {
    case PageAccess::kNoAccess:
      return PROT_NONE;
    case PageAccess::kReadOnly:
      return PROT_READ;
    case PageAccess::kReadWrite:
      return PROT_READ | PROT_WRITE;
    case PageAccess::kExecuteReadOnly:
      return PROT_READ | PROT_EXEC;
    case PageAccess::kExecuteReadWrite:
      return PROT_READ | PROT_WRITE | PROT_EXEC;
    default:
      assert_unhandled_case(access);
      return PROT_NONE;
  }
}

bool IsWritableExecutableMemorySupported() {
  static const bool supported = []() {
    const size_t test_size = page_size();
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if XE_PLATFORM_MAC && defined(__aarch64__)
#ifdef MAP_JIT
    // MAP_JIT is required for executable mappings on macOS ARM64.
    flags |= MAP_JIT;
#endif
#endif
    void* test_mapping = mmap(nullptr, test_size,
                              PROT_READ | PROT_WRITE | PROT_EXEC, flags, -1, 0);
    if (test_mapping == MAP_FAILED) {
      return false;
    }
    munmap(test_mapping, test_size);
    return true;
  }();
  return supported;
}

void* AllocFixed(void* base_address, size_t length,
                 AllocationType allocation_type, PageAccess access) {
  // mmap does not support reserve / commit, so ignore allocation_type.
  uint32_t prot = ToPosixProtectFlags(access);
  if (base_address && allocation_type == AllocationType::kCommit) {
    const size_t system_page_size = page_size();
    uintptr_t start = reinterpret_cast<uintptr_t>(base_address);
    uintptr_t aligned_start = start & ~(system_page_size - 1);
    uintptr_t aligned_end =
        xe::align(start + length, system_page_size);
    size_t aligned_length =
        aligned_end > aligned_start ? aligned_end - aligned_start : 0;
    if (!aligned_length) {
      return base_address;
    }
    return mprotect(reinterpret_cast<void*>(aligned_start), aligned_length,
                    prot) == 0
               ? base_address
               : nullptr;
  }

#if XE_PLATFORM_MAC && defined(__aarch64__)
  // On macOS ARM64, MAP_JIT is required for executable mappings.
  uint32_t flags = MAP_PRIVATE | MAP_ANONYMOUS;
  if (access == PageAccess::kExecuteReadWrite ||
      access == PageAccess::kExecuteReadOnly) {
    flags |= MAP_JIT;
  }

  // Align the requested base address to the system page size if provided.
  uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(base_address);
  if (aligned_addr != 0) {
    aligned_addr = xe::round_up(aligned_addr, page_size());
  }

  uint32_t fixed_flags = flags;
#ifdef MAP_FIXED_NOREPLACE
  if (aligned_addr != 0) {
    fixed_flags |= MAP_FIXED_NOREPLACE;
  }
#endif
  void* result = mmap(aligned_addr ? reinterpret_cast<void*>(aligned_addr)
                                   : nullptr,
                      length, prot, aligned_addr ? fixed_flags : flags, -1, 0);
  if (result == MAP_FAILED && aligned_addr != 0) {
    if (errno == EINVAL) {
      result = mmap(reinterpret_cast<void*>(aligned_addr), length, prot, flags,
                    -1, 0);
    }
  }
  if (result == MAP_FAILED) {
    return nullptr;
  }
  if (aligned_addr != 0 &&
      result != reinterpret_cast<void*>(aligned_addr)) {
    munmap(result, length);
    return nullptr;
  }
  return result;
#else
  uint32_t flags = MAP_PRIVATE | MAP_ANONYMOUS;
  uint32_t fixed_flags = flags;
#ifdef MAP_FIXED_NOREPLACE
  if (base_address) {
    fixed_flags |= MAP_FIXED_NOREPLACE;
  }
#endif
  void* result =
      mmap(base_address, length, prot,
           base_address ? fixed_flags : flags, -1, 0);
  if (result == MAP_FAILED && base_address) {
    if (errno == EINVAL) {
      result = mmap(base_address, length, prot, flags, -1, 0);
    }
  }
  if (result == MAP_FAILED) {
    return nullptr;
  }
  if (base_address && result != base_address) {
    munmap(result, length);
    return nullptr;
  }
  return result;
#endif
}

bool DeallocFixed(void* base_address, size_t length,
                  DeallocationType deallocation_type) {
  return munmap(base_address, length) == 0;
}

bool Protect(void* base_address, size_t length, PageAccess access,
             PageAccess* out_old_access) {
  // Linux does not have a syscall to query memory permissions.
  assert_null(out_old_access);

  uint32_t prot = ToPosixProtectFlags(access);
  return mprotect(base_address, length, prot) == 0;
}

bool QueryProtect(void* base_address, size_t& length, PageAccess& access_out) {
  return false;
}

FileMappingHandle CreateFileMappingHandle(const std::filesystem::path& path,
                                          size_t length, PageAccess access,
                                          bool commit) {
#if XE_PLATFORM_ANDROID
  // TODO(Triang3l): Check if memfd can be used instead on API 30+.
  if (android_ASharedMemory_create_) {
    int sharedmem_fd = android_ASharedMemory_create_(path.c_str(), length);
    if (sharedmem_fd >= 0 && android_ASharedMemory_setProt_) {
      android_ASharedMemory_setProt_(sharedmem_fd,
                                     ToPosixProtectFlags(access));
    }
    return sharedmem_fd >= 0 ? sharedmem_fd : kFileMappingHandleInvalid;
  }

  // Use /dev/ashmem on API versions below 26, which added ASharedMemory.
  // /dev/ashmem was disabled on API 29 for apps targeting it.
  // https://chromium.googlesource.com/chromium/src/+/master/third_party/ashmem/ashmem-dev.c
  int ashmem_fd = open("/" ASHMEM_NAME_DEF, O_RDWR);
  if (ashmem_fd < 0) {
    return kFileMappingHandleInvalid;
  }
  char ashmem_name[ASHMEM_NAME_LEN];
  strlcpy(ashmem_name, path.c_str(), xe::countof(ashmem_name));
  if (ioctl(ashmem_fd, ASHMEM_SET_NAME, ashmem_name) < 0 ||
      ioctl(ashmem_fd, ASHMEM_SET_SIZE, length) < 0) {
    close(ashmem_fd);
    return kFileMappingHandleInvalid;
  }
  return ashmem_fd;
#else
  int oflag;
  switch (access) {
    case PageAccess::kNoAccess:
      oflag = 0;
      break;
    case PageAccess::kReadOnly:
    case PageAccess::kExecuteReadOnly:
      oflag = O_RDONLY;
      break;
    case PageAccess::kReadWrite:
    case PageAccess::kExecuteReadWrite:
      oflag = O_RDWR;
      break;
    default:
      assert_always();
      return kFileMappingHandleInvalid;
  }
  oflag |= O_CREAT;
  auto full_path = "/" / path;
  int ret = shm_open(full_path.c_str(), oflag, 0777);
  if (ret < 0) {
    return kFileMappingHandleInvalid;
  }
#ifdef __APPLE__
    ftruncate(ret, length);
#else
  ftruncate64(ret, length);
#endif
  return ret;
#endif
}

void CloseFileMappingHandle(FileMappingHandle handle,
                            const std::filesystem::path& path) {
  close(handle);
#if !XE_PLATFORM_ANDROID
  auto full_path = "/" / path;
  shm_unlink(full_path.c_str());
#endif
}

void* MapFileView(FileMappingHandle handle, void* base_address, size_t length,
                  PageAccess access, size_t file_offset) {
  uint32_t prot = ToPosixProtectFlags(access);
  uint32_t flags = MAP_SHARED;
  uint32_t fixed_flags = flags;
#ifdef MAP_FIXED_NOREPLACE
  if (base_address) {
    fixed_flags |= MAP_FIXED_NOREPLACE;
  }
#endif
#ifdef __APPLE__
  void* result = mmap(base_address, length, prot,
                      base_address ? fixed_flags : flags, handle, file_offset);
#else
  void* result = mmap64(base_address, length, prot,
                        base_address ? fixed_flags : flags, handle,
                        file_offset);
#endif
  if (result == MAP_FAILED && base_address) {
    if (errno == EINVAL) {
#ifdef __APPLE__
      result = mmap(base_address, length, prot, flags, handle, file_offset);
#else
      result = mmap64(base_address, length, prot, flags, handle, file_offset);
#endif
    }
  }
  if (result == MAP_FAILED) {
    return nullptr;
  }
  if (base_address && result != base_address) {
    munmap(result, length);
    return nullptr;
  }
  return result;
}


bool UnmapFileView(FileMappingHandle handle, void* base_address,
                   size_t length) {
  return munmap(base_address, length) == 0;
}

}  // namespace memory
}  // namespace xe
