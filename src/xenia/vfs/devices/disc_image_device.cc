/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/vfs/devices/disc_image_device.h"

#include <cstring>
#include <vector>

#include "xenia/base/literals.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/vfs/devices/disc_image_entry.h"
#include "xenia/vfs/gdfx_util.h"

namespace xe {
namespace vfs {

using namespace xe::literals;

DiscImageDevice::DiscImageDevice(const std::string_view mount_path,
                                 const std::filesystem::path& host_path)
    : Device(mount_path), name_("GDFX"), host_path_(host_path) {}

DiscImageDevice::~DiscImageDevice() = default;

bool DiscImageDevice::Initialize() {
  mmap_ = MappedMemory::Open(host_path_, MappedMemory::Mode::kRead);
  if (!mmap_) {
#if XE_PLATFORM_IOS
    file_handle_ = xe::filesystem::FileHandle::OpenExisting(
        host_path_, xe::filesystem::FileAccess::kGenericRead);
    auto file_info = xe::filesystem::GetInfo(host_path_);
    if (!file_handle_ || !file_info ||
        file_info->type != xe::filesystem::FileInfo::Type::kFile) {
      XELOGE("Disc image could not be opened for fallback reads");
      return false;
    }
    image_size_ = file_info->total_size;
    XELOGW(
        "Disc image mmap unavailable, using iOS fallback file reads "
        "(size=0x{:X})",
        image_size_);
#else
    XELOGE("Disc image could not be mapped");
    return false;
#endif  // XE_PLATFORM_IOS
  } else {
    image_size_ = mmap_->size();
    XELOGFS("DiscImageDevice::Initialize");
  }

  ParseState state = {0};
  state.ptr = mmap_ ? mmap_->data() : nullptr;
  state.size = image_size_;
  auto result = Verify(&state);
  if (result != Error::kSuccess) {
    XELOGE("Failed to verify disc image header: {}",
           static_cast<int32_t>(result));
    return false;
  }

  if (state.root_offset > state.size ||
      state.root_size > (state.size - state.root_offset)) {
    XELOGE("Disc image root directory is out of bounds");
    return false;
  }

  std::vector<uint8_t> root_buffer_storage;
  const uint8_t* root_buffer = nullptr;
  if (state.ptr) {
    root_buffer = state.ptr + state.root_offset;
  } else {
    root_buffer_storage.resize(state.root_size);
    if (!ReadImage(state.root_offset, root_buffer_storage.data(),
                   root_buffer_storage.size())) {
      XELOGE("Failed to read disc image root directory");
      return false;
    }
    root_buffer = root_buffer_storage.data();
  }

  result = ReadAllEntries(&state, root_buffer);
  if (result != Error::kSuccess) {
    XELOGE("Failed to read all GDFX entries: {}", static_cast<int32_t>(result));
    return false;
  }

  return true;
}

void DiscImageDevice::Dump(StringBuffer* string_buffer) {
  auto global_lock = global_critical_region_.Acquire();
  root_entry_->Dump(string_buffer, 0);
}

Entry* DiscImageDevice::ResolvePath(const std::string_view path) {
  // The filesystem will have stripped our prefix off already, so the path will
  // be in the form:
  // some\PATH.foo
  XELOGFS("DiscImageDevice::ResolvePath({})", path);
  return root_entry_->ResolvePath(path);
}

DiscImageDevice::Error DiscImageDevice::Verify(ParseState* state) {
  if (!state->ptr) {
    for (size_t offset : kGdfxLikelyOffsets) {
      const size_t sector32_offset = offset + (32 * kGdfxSectorSize);
      if (sector32_offset > state->size || state->size - sector32_offset < 28) {
        continue;
      }
      if (!VerifyMagic(state, sector32_offset)) {
        continue;
      }

      uint8_t fs_data[28] = {};
      if (!ReadImage(sector32_offset, fs_data, sizeof(fs_data))) {
        continue;
      }

      uint32_t root_sector = xe::load<uint32_t>(fs_data + 20);
      uint32_t root_size = xe::load<uint32_t>(fs_data + 24);
      if (root_size < 13 || root_size > 32 * 1024 * 1024) {
        continue;
      }

      state->game_offset = offset;
      state->root_sector = root_sector;
      state->root_size = root_size;
      state->root_offset =
          state->game_offset + (state->root_sector * kGdfxSectorSize);
      return Error::kSuccess;
    }
    return Error::kErrorFileMismatch;
  }

  // Use shared GDFX utility to find the game partition.
  auto partition = GdfxFindPartition(state->ptr, state->size);
  if (!partition) {
    return Error::kErrorFileMismatch;
  }

  state->game_offset = partition->game_offset;
  state->root_sector = partition->root_sector;
  state->root_size = partition->root_size;
  state->root_offset =
      state->game_offset + (state->root_sector * kGdfxSectorSize);

  return Error::kSuccess;
}

bool DiscImageDevice::VerifyMagic(ParseState* state, size_t offset) {
  if (!state->ptr) {
    uint8_t magic[kGdfxMagicSize] = {};
    if (!ReadImage(offset, magic, kGdfxMagicSize)) {
      return false;
    }
    return std::memcmp(magic, kGdfxMagic, kGdfxMagicSize) == 0;
  }
  return GdfxVerifyMagic(state->ptr, state->size, offset);
}

DiscImageDevice::Error DiscImageDevice::ReadAllEntries(
    ParseState* state, const uint8_t* root_buffer) {
  auto root_entry = new DiscImageEntry(this, nullptr, "", mmap_.get());
  root_entry->attributes_ = kFileAttributeDirectory;
  root_entry_ = std::unique_ptr<Entry>(root_entry);

  if (!ReadEntry(state, root_buffer, 0, root_entry)) {
    return Error::kErrorOutOfMemory;
  }

  return Error::kSuccess;
}

bool DiscImageDevice::ReadEntry(ParseState* state, const uint8_t* buffer,
                                uint16_t entry_ordinal,
                                DiscImageEntry* parent) {
  const uint8_t* p = buffer + (entry_ordinal * 4);

  uint16_t node_l = xe::load<uint16_t>(p + 0);
  uint16_t node_r = xe::load<uint16_t>(p + 2);
  size_t sector = xe::load<uint32_t>(p + 4);
  size_t length = xe::load<uint32_t>(p + 8);
  uint8_t attributes = xe::load<uint8_t>(p + 12);
  uint8_t name_length = xe::load<uint8_t>(p + 13);
  auto name_buffer = reinterpret_cast<const char*>(p + 14);

  if (node_l && !ReadEntry(state, buffer, node_l, parent)) {
    return false;
  }

  // Filename is stored as Windows-1252, convert it to UTF-8.
  auto ansi_name = std::string(name_buffer, name_length);
  auto name = xe::win1252_to_utf8(ansi_name);
  // Fallback to normal name if for whatever reason conversion from 1252 code
  // page failed.
  if (name.empty()) {
    name = ansi_name;
  }

  auto entry = DiscImageEntry::Create(this, parent, name, mmap_.get());
  entry->attributes_ = attributes | kFileAttributeReadOnly;
  entry->size_ = length;
  entry->allocation_size_ = xe::round_up(length, bytes_per_sector());

  // Set to January 1, 1970 (UTC) in 100-nanosecond intervals
  entry->create_timestamp_ = 10000 * 11644473600000LL;
  entry->access_timestamp_ = 10000 * 11644473600000LL;
  entry->write_timestamp_ = 10000 * 11644473600000LL;

  if (attributes & kFileAttributeDirectory) {
    // Folder.
    entry->data_offset_ = 0;
    entry->data_size_ = 0;
    if (length) {
      // Not a leaf - read in children.
      const size_t folder_offset =
          state->game_offset + (sector * kGdfxSectorSize);
      if (folder_offset > state->size ||
          length > (state->size - folder_offset)) {
        // Out of bounds read.
        return false;
      }
      if (state->ptr) {
        // Read child list directly from mapped memory.
        uint8_t* folder_ptr = state->ptr + folder_offset;
        if (!ReadEntry(state, folder_ptr, 0, entry.get())) {
          return false;
        }
      } else {
        std::vector<uint8_t> folder_data(length);
        if (!ReadImage(folder_offset, folder_data.data(), folder_data.size())) {
          return false;
        }
        if (!ReadEntry(state, folder_data.data(), 0, entry.get())) {
          return false;
        }
      }
    }
  } else {
    // File.
    entry->data_offset_ = state->game_offset + (sector * kGdfxSectorSize);
    entry->data_size_ = length;
  }

  // Add to parent.
  parent->children_.emplace_back(std::move(entry));

  // Read next file in the list.
  if (node_r && !ReadEntry(state, buffer, node_r, parent)) {
    return false;
  }

  return true;
}

bool DiscImageDevice::ReadImage(size_t offset, void* buffer,
                                size_t length) const {
  if (!length) {
    return true;
  }
  if (offset > image_size_ || length > (image_size_ - offset)) {
    return false;
  }
  if (mmap_) {
    std::memcpy(buffer, mmap_->data() + offset, length);
    return true;
  }
  if (!file_handle_) {
    return false;
  }
  size_t bytes_read = 0;
  if (!file_handle_->Read(offset, buffer, length, &bytes_read)) {
    return false;
  }
  return bytes_read == length;
}

}  // namespace vfs
}  // namespace xe
