/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/lzx.h"

#include <algorithm>
#include <climits>
#include <cstring>

#include "xenia/base/byte_order.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/kernel/util/xex2_info.h"

#include "third_party/mspack/lzx.h"
#include "third_party/mspack/mspack.h"

typedef struct mspack_memory_file_t {
  mspack_system sys;
  void* buffer;
  off_t buffer_size;
  off_t offset;
} mspack_memory_file;

mspack_memory_file* mspack_memory_open(mspack_system* sys, void* buffer,
                                       const size_t buffer_size) {
  assert_true(buffer_size < INT_MAX);
  if (buffer_size >= INT_MAX) {
    return NULL;
  }
  auto memfile =
      (mspack_memory_file*)std::calloc(1, sizeof(mspack_memory_file));
  if (!memfile) {
    return NULL;
  }
  memfile->buffer = buffer;
  memfile->buffer_size = (off_t)buffer_size;
  memfile->offset = 0;
  return memfile;
}

void mspack_memory_close(mspack_memory_file* file) {
  auto memfile = (mspack_memory_file*)file;
  std::free(memfile);
}

int mspack_memory_read(mspack_file* file, void* buffer, int chars) {
  auto memfile = (mspack_memory_file*)file;
  const off_t remaining = memfile->buffer_size - memfile->offset;
  const off_t total = std::min(static_cast<off_t>(chars), remaining);
  std::memcpy(buffer, (uint8_t*)memfile->buffer + memfile->offset, total);
  memfile->offset += total;
  return (int)total;
}

int mspack_memory_write(mspack_file* file, void* buffer, int chars) {
  auto memfile = (mspack_memory_file*)file;
  const off_t remaining = memfile->buffer_size - memfile->offset;
  const off_t total = std::min(static_cast<off_t>(chars), remaining);
  std::memcpy((uint8_t*)memfile->buffer + memfile->offset, buffer, total);
  memfile->offset += total;
  return (int)total;
}

void* mspack_memory_alloc(mspack_system* sys, size_t chars) {
  return std::calloc(chars, 1);
}

void mspack_memory_free(void* ptr) { std::free(ptr); }

void mspack_memory_copy(void* src, void* dest, size_t chars) {
  std::memcpy(dest, src, chars);
}

mspack_system* mspack_memory_sys_create() {
  auto sys = (mspack_system*)std::calloc(1, sizeof(mspack_system));
  if (!sys) {
    return NULL;
  }
  sys->read = mspack_memory_read;
  sys->write = mspack_memory_write;
  sys->alloc = mspack_memory_alloc;
  sys->free = mspack_memory_free;
  sys->copy = mspack_memory_copy;
  return sys;
}

void mspack_memory_sys_destroy(struct mspack_system* sys) { free(sys); }

int lzx_decompress(const void* lzx_data, size_t lzx_len, void* dest,
                   size_t dest_len, uint32_t window_size, void* window_data,
                   size_t window_data_len) {
  int result_code = 1;

  uint32_t window_bits;
  if (!xe::bit_scan_forward(window_size, &window_bits)) {
    return result_code;
  }

  mspack_system* sys = mspack_memory_sys_create();
  mspack_memory_file* lzxsrc =
      mspack_memory_open(sys, (void*)lzx_data, lzx_len);
  mspack_memory_file* lzxdst = mspack_memory_open(sys, dest, dest_len);
  lzxd_stream* lzxd = lzxd_init(sys, (mspack_file*)lzxsrc, (mspack_file*)lzxdst,
                                window_bits, 0, 0x8000, (off_t)dest_len, 0);

  if (lzxd) {
    if (window_data) {
      // zero the window and then copy window_data to the end of it
      auto padding_len = window_size - window_data_len;
      std::memset(&lzxd->window[0], 0, padding_len);
      std::memcpy(&lzxd->window[padding_len], window_data, window_data_len);
      // TODO(gibbed): should this be set regardless if source window data is
      // available or not?
      lzxd->ref_data_size = window_size;
    }

    result_code = lzxd_decompress(lzxd, (off_t)dest_len);

    lzxd_free(lzxd);
    lzxd = NULL;
  }

  if (lzxsrc) {
    mspack_memory_close(lzxsrc);
    lzxsrc = NULL;
  }

  if (lzxdst) {
    mspack_memory_close(lzxdst);
    lzxdst = NULL;
  }

  if (sys) {
    mspack_memory_sys_destroy(sys);
    sys = NULL;
  }

  return result_code;
}

int lzxdelta_apply_patch(xe::xex2_delta_patch* patch, size_t patch_len,
                         uint32_t window_size, void* dest) {
  const uint8_t* cur = reinterpret_cast<const uint8_t*>(patch);
  const uint8_t* end = cur + patch_len;

  auto read_u32be = [](const uint8_t* p) -> uint32_t {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return xe::byte_swap(v);
  };
  auto read_u16be = [](const uint8_t* p) -> uint16_t {
    uint16_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return xe::byte_swap(v);
  };

  // Patch entry header is 12 bytes (old/new addr + uncompressed/compressed len).
  while (cur + 12 <= end) {
    const uint32_t old_addr = read_u32be(cur + 0);
    const uint32_t new_addr = read_u32be(cur + 4);
    const uint16_t uncompressed_len = read_u16be(cur + 8);
    const uint16_t compressed_len = read_u16be(cur + 10);

    if (compressed_len == 0 && uncompressed_len == 0 && new_addr == 0 &&
        old_addr == 0) {
      break;
    }

    const uint8_t* patch_data = cur + 12;
    size_t entry_size = 12;
    if (compressed_len > 1) {
      entry_size += compressed_len;
      if (cur + entry_size > end) {
        XELOGE("LZX delta patch truncated (need {} bytes, have {}).",
               entry_size, static_cast<size_t>(end - cur));
        return 1;
      }
    }

    switch (compressed_len) {
      case 0:  // fill with 0
        std::memset(reinterpret_cast<char*>(dest) + new_addr, 0,
                    uncompressed_len);
        break;
      case 1:  // copy from old -> new (overlap allowed)
        std::memmove(reinterpret_cast<char*>(dest) + new_addr,
                     reinterpret_cast<char*>(dest) + old_addr,
                     uncompressed_len);
        break;
      default: {  // delta patch
        int result = lzx_decompress(
            patch_data, compressed_len,
            reinterpret_cast<char*>(dest) + new_addr, uncompressed_len,
            window_size, reinterpret_cast<char*>(dest) + old_addr,
            uncompressed_len);
        if (result) {
          return result;
        }
        break;
      }
    }

    cur += entry_size;
  }

  return 0;
}
