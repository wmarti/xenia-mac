/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/vfs/virtual_file_system.h"

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/string.h"
#include "xenia/kernel/xfile.h"

namespace xe {
namespace vfs {

DEFINE_bool(log_cache_open_failures, false,
            "Log cache0:/cache1:/cache: open failures in VFS.", "VFS");
DEFINE_bool(log_cache_resolve_failures, false,
            "Log cache0/cache1/cache ResolvePath misses in VFS.", "VFS");
DEFINE_bool(enable_relative_path_fallback, false,
            "Fallback to game:\\ for relative guest paths.", "VFS");
DEFINE_bool(enable_cache_path_remap, false,
            "Remap \\Device\\Harddisk0\\Cache* to \\CACHE*.", "VFS");

namespace {

bool IsGuestAbsolutePath(const std::string_view path) {
  if (path.empty()) {
    return false;
  }
  if (path.find(':') != std::string_view::npos) {
    return true;
  }
  return xe::utf8::starts_with_case(path, "\\Device\\") ||
         xe::utf8::starts_with_case(path, "\\??\\") ||
         xe::utf8::starts_with_case(path, "\\GLOBAL??\\") ||
         xe::utf8::starts_with_case(path, "\\CACHE0") ||
         xe::utf8::starts_with_case(path, "\\CACHE1") ||
         xe::utf8::starts_with_case(path, "\\CACHE");
}

std::string MakeGameRelativePath(const std::string_view path) {
  if (path.empty()) {
    return std::string();
  }
  if (xe::utf8::starts_with_case(path, "game:")) {
    return std::string(path);
  }
  if (path.front() == '\\' || path.front() == '/') {
    return std::string("game:") + std::string(path);
  }
  return std::string("game:\\") + std::string(path);
}

}  // namespace

VirtualFileSystem::VirtualFileSystem() {}

VirtualFileSystem::~VirtualFileSystem() {
  // Delete all devices.
  // This will explode if anyone is still using data from them.
  devices_.clear();
  symlinks_.clear();
}

bool VirtualFileSystem::RegisterDevice(std::unique_ptr<Device> device) {
  auto global_lock = global_critical_region_.Acquire();
  devices_.emplace_back(std::move(device));
  return true;
}

bool VirtualFileSystem::UnregisterDevice(const std::string_view path) {
  auto global_lock = global_critical_region_.Acquire();
  for (auto it = devices_.begin(); it != devices_.end(); ++it) {
    if ((*it)->mount_path() == path) {
      XELOGD("Unregistered device: {}", (*it)->mount_path());
      devices_.erase(it);
      return true;
    }
  }
  return false;
}

bool VirtualFileSystem::RegisterSymbolicLink(const std::string_view path,
                                             const std::string_view target) {
  auto global_lock = global_critical_region_.Acquire();
  symlinks_.insert({std::string(path), std::string(target)});
  XELOGD("Registered symbolic link: {} => {}", path, target);

  return true;
}

bool VirtualFileSystem::UnregisterSymbolicLink(const std::string_view path) {
  auto global_lock = global_critical_region_.Acquire();
  auto it = std::find_if(
      symlinks_.cbegin(), symlinks_.cend(),
      [&](const auto& s) { return xe::utf8::equal_case(path, s.first); });
  if (it == symlinks_.end()) {
    return false;
  }
  XELOGD("Unregistered symbolic link: {} => {}", it->first, it->second);

  symlinks_.erase(it);
  return true;
}

bool VirtualFileSystem::FindSymbolicLink(const std::string_view path,
                                         std::string& target) {
  auto it = std::find_if(
      symlinks_.cbegin(), symlinks_.cend(),
      [&](const auto& s) { return xe::utf8::starts_with_case(path, s.first); });
  if (it == symlinks_.cend()) {
    return false;
  }
  target = (*it).second;
  return true;
}

bool VirtualFileSystem::ResolveSymbolicLink(const std::string_view path,
                                            std::string& result) {
  result = path;
  bool was_resolved = false;
  const bool log_cache_symlink =
      cvars::log_cache_resolve_failures &&
      xe::utf8::starts_with_case(result, "\\Device\\Harddisk0\\Cache");
  if (log_cache_symlink) {
    XELOGW("ResolveSymbolicLink cache: input={}", result);
  }
  while (true) {
    auto it =
        std::find_if(symlinks_.cbegin(), symlinks_.cend(), [&](const auto& s) {
          return xe::utf8::starts_with_case(result, s.first);
        });
    if (it == symlinks_.cend()) {
      break;
    }
    // Found symlink!
    auto target_path = (*it).second;
    auto relative_path = result.substr((*it).first.size());
    result = target_path + relative_path;
    was_resolved = true;
    if (log_cache_symlink) {
      XELOGW("ResolveSymbolicLink cache: matched {} => {} (relative={})",
             (*it).first, target_path, relative_path);
    }
  }
  if (log_cache_symlink && was_resolved) {
    XELOGW("ResolveSymbolicLink cache: output={}", result);
  }
  return was_resolved;
}

Entry* VirtualFileSystem::ResolvePath(const std::string_view path) {
  auto global_lock = global_critical_region_.Acquire();

  // Resolve relative paths
  auto normalized_path(xe::utf8::canonicalize_guest_path(path));

  // Resolve symlinks.
  std::string resolved_path;
  if (ResolveSymbolicLink(normalized_path, resolved_path)) {
    normalized_path = resolved_path;
  }

  const auto resolve_with_candidate =
      [&](const std::string& candidate) -> Entry* {
    auto it = std::find_if(
        devices_.cbegin(), devices_.cend(), [&](const auto& d) {
          return xe::utf8::starts_with(candidate, d->mount_path());
        });
    if (it == devices_.cend()) {
      return nullptr;
    }
    const auto& device = *it;
    auto relative_path = candidate.substr(device->mount_path().size());
    auto entry = device->ResolvePath(relative_path);
    if (!entry && cvars::log_cache_resolve_failures) {
      const auto& mount_path = device->mount_path();
      if (xe::utf8::equal_case(mount_path, "\\CACHE0") ||
          xe::utf8::equal_case(mount_path, "\\CACHE1") ||
          xe::utf8::equal_case(mount_path, "\\CACHE")) {
        XELOGW("Cache ResolvePath miss: path={} rel={}", candidate,
               relative_path);
      }
    }
    return entry;
  };

  if (auto entry = resolve_with_candidate(normalized_path)) {
    return entry;
  }

  if (cvars::enable_relative_path_fallback &&
      !IsGuestAbsolutePath(normalized_path)) {
    auto fallback_path = MakeGameRelativePath(normalized_path);
    std::string resolved_fallback;
    if (ResolveSymbolicLink(fallback_path, resolved_fallback)) {
      fallback_path = resolved_fallback;
    }
    if (auto entry = resolve_with_candidate(fallback_path)) {
      return entry;
    }
  }

  // Supress logging the error for ShaderDumpxe:\CompareBackEnds as this is
  // not an actual problem nor something we care about.
  if (path != "ShaderDumpxe:\\CompareBackEnds") {
    XELOGE("ResolvePath({}) failed - device not found", path);
  }
  return nullptr;
}

Entry* VirtualFileSystem::CreatePath(const std::string_view path,
                                     uint32_t attributes) {
  // Create all required directories recursively.
  auto path_parts = xe::utf8::split_path(path);
  if (path_parts.empty()) {
    return nullptr;
  }
  auto partial_path = std::string(path_parts[0]);
  auto partial_entry = ResolvePath(partial_path);
  if (!partial_entry) {
    return nullptr;
  }
  auto parent_entry = partial_entry;
  for (size_t i = 1; i < path_parts.size() - 1; ++i) {
    partial_path = xe::utf8::join_guest_paths(partial_path, path_parts[i]);
    auto child_entry = ResolvePath(partial_path);
    if (!child_entry) {
      child_entry =
          parent_entry->CreateEntry(path_parts[i], kFileAttributeDirectory);
    }
    if (!child_entry) {
      return nullptr;
    }
    parent_entry = child_entry;
  }
  return parent_entry->CreateEntry(path_parts[path_parts.size() - 1],
                                   attributes);
}

bool VirtualFileSystem::DeletePath(const std::string_view path) {
  auto entry = ResolvePath(path);
  if (!entry) {
    return false;
  }
  auto parent = entry->parent();
  if (!parent) {
    // Can't delete root.
    return false;
  }
  return parent->Delete(entry);
}

X_STATUS VirtualFileSystem::OpenFile(Entry* root_entry,
                                     const std::string_view path,
                                     FileDisposition creation_disposition,
                                     uint32_t desired_access, bool is_directory,
                                     bool is_non_directory, File** out_file,
                                     FileAction* out_action) {
  // TODO(gibbed): should 'is_directory' remain as a bool or should it be
  // flipped to a generic FileAttributeFlags?

  // Cleanup access.
  if (desired_access & FileAccess::kGenericRead) {
    desired_access |= FileAccess::kFileReadData;
  }
  if (desired_access & FileAccess::kGenericWrite) {
    desired_access |= FileAccess::kFileWriteData;
  }
  if (desired_access & FileAccess::kGenericAll) {
    desired_access |= FileAccess::kFileReadData | FileAccess::kFileWriteData;
  }

  std::string resolved_path(path);
  const auto has_device = [&](const std::string_view mount_path) {
    return std::any_of(devices_.cbegin(), devices_.cend(),
                       [&](const auto& device) {
                         return xe::utf8::equal_case(device->mount_path(),
                                                     mount_path);
                       });
  };

  const auto remap_harddisk_cache =
      [&](const std::string_view prefix,
          const std::string_view target_mount) -> bool {
    if (!xe::utf8::starts_with_case(resolved_path, prefix)) {
      return false;
    }
    if (!has_device(target_mount)) {
      return false;
    }
    const auto suffix = resolved_path.substr(prefix.size());
    resolved_path = std::string(target_mount) + suffix;
    return true;
  };

  if (cvars::enable_cache_path_remap) {
    remap_harddisk_cache("\\Device\\Harddisk0\\Cache0", "\\CACHE0");
    remap_harddisk_cache("\\Device\\Harddisk0\\Cache1", "\\CACHE1");
    remap_harddisk_cache("\\Device\\Harddisk0\\Cache", "\\CACHE");
    if (root_entry && root_entry->device() &&
        xe::utf8::equal_case(root_entry->device()->mount_path(),
                             "\\Device\\Harddisk0")) {
      if (xe::utf8::starts_with_case(resolved_path, "\\Cache0") ||
          xe::utf8::starts_with_case(resolved_path, "\\Cache1") ||
          xe::utf8::starts_with_case(resolved_path, "\\Cache")) {
        // Some titles pass absolute cache paths with a Harddisk0 root handle.
        // Route those to the cache devices instead of resolving via NullDevice.
        root_entry = nullptr;
      }
    }
  }

  // Lookup host device/parent path.
  // If no device or parent, fail.
  Entry* parent_entry = nullptr;
  Entry* entry = nullptr;

  auto base_path = xe::utf8::find_base_guest_path(resolved_path);
  if (!base_path.empty()) {
    parent_entry = !root_entry ? ResolvePath(base_path)
                               : root_entry->ResolvePath(base_path);
    if (!parent_entry) {
      *out_action = FileAction::kDoesNotExist;
      return X_STATUS_NO_SUCH_FILE;
    }

    auto file_name = xe::utf8::find_name_from_guest_path(resolved_path);
    entry = parent_entry->GetChild(file_name);
  } else {
    entry =
        !root_entry ? ResolvePath(resolved_path) : root_entry->GetChild(path);
  }

  if (entry) {
    if (entry->attributes() & kFileAttributeDirectory && is_non_directory) {
      return X_STATUS_FILE_IS_A_DIRECTORY;
    }
  }

  // Check if exists (if we need it to), or that it doesn't (if it shouldn't).
  const auto is_cache_path = [](const std::string_view p) {
    return xe::utf8::starts_with_case(p, "cache0:") ||
           xe::utf8::starts_with_case(p, "cache1:") ||
           xe::utf8::starts_with_case(p, "cache:") ||
           xe::utf8::starts_with_case(p, "\\\\CACHE0") ||
           xe::utf8::starts_with_case(p, "\\\\CACHE1") ||
           xe::utf8::starts_with_case(p, "\\\\CACHE");
  };

  switch (creation_disposition) {
    case FileDisposition::kOpen:
    case FileDisposition::kOverwrite:
      // Must exist.
      if (!entry) {
        if (cvars::log_cache_open_failures && is_cache_path(path)) {
          XELOGW(
              "Cache OpenFile miss: path={} disp={} access={:08X} is_dir={} "
              "is_non_dir={}",
              path, static_cast<uint32_t>(creation_disposition), desired_access,
              is_directory, is_non_directory);
        }
        *out_action = FileAction::kDoesNotExist;
        return X_STATUS_NO_SUCH_FILE;
      }
      break;
    case FileDisposition::kCreate:
      // Must not exist.
      if (entry) {
        *out_action = FileAction::kExists;
        return X_STATUS_OBJECT_NAME_COLLISION;
      }
      break;
    default:
      // Either way, ok.
      break;
  }

  // Verify permissions.
  bool wants_write = desired_access & FileAccess::kFileWriteData ||
                     desired_access & FileAccess::kFileAppendData;
  if (wants_write && ((parent_entry && parent_entry->is_read_only()) ||
                      (entry && entry->is_read_only()))) {
    // Fail if read only device and wants write.
    // return X_STATUS_ACCESS_DENIED;
    // TODO(benvanik): figure out why games are opening read-only files with
    // write modes.
    assert_always();
    XELOGW("Attempted to open the file/dir for create/write");
    desired_access = FileAccess::kGenericRead | FileAccess::kFileReadData;
  }

  bool created = false;
  if (!entry) {
    // Remember that we are creating this new, instead of replacing.
    created = true;
    *out_action = FileAction::kCreated;
  } else {
    // May need to delete, if it exists.
    switch (creation_disposition) {
      case FileDisposition::kCreate:
        // Shouldn't be possible to hit this.
        assert_always();
        return X_STATUS_ACCESS_DENIED;
      case FileDisposition::kSuperscede:
        // Replace (by delete + recreate).
        if (!entry->Delete()) {
          return X_STATUS_ACCESS_DENIED;
        }
        entry = nullptr;
        *out_action = FileAction::kSuperseded;
        break;
      case FileDisposition::kOpen:
      case FileDisposition::kOpenIf:
        // Normal open.
        *out_action = FileAction::kOpened;
        break;
      case FileDisposition::kOverwrite:
      case FileDisposition::kOverwriteIf:
        // Overwrite (we do by delete + recreate).
        if (!entry->Delete()) {
          return X_STATUS_ACCESS_DENIED;
        }
        entry = nullptr;
        *out_action = FileAction::kOverwritten;
        break;
    }
  }
  if (!entry) {
    // Create if needed (either new or as a replacement).
    entry = CreatePath(
        path, !is_directory ? kFileAttributeNormal : kFileAttributeDirectory);
    if (!entry) {
      return X_STATUS_ACCESS_DENIED;
    }
  }

  // Open.
  auto result = entry->Open(desired_access, out_file);
  if (XFAILED(result)) {
    if (cvars::log_cache_open_failures && is_cache_path(path)) {
      XELOGW(
          "Cache OpenFile failed: path={} disp={} access={:08X} is_dir={} "
          "is_non_dir={} status={:08X}",
          path, static_cast<uint32_t>(creation_disposition), desired_access,
          is_directory, is_non_directory, result);
    }
    *out_action = FileAction::kDoesNotExist;
  }
  return result;
}

}  // namespace vfs
}  // namespace xe
