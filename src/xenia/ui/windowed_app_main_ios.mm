/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <TargetConditionals.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/string.h"
#include "xenia/config.h"
#include "xenia/ui/surface_ios.h"
#include "xenia/ui/windowed_app.h"
#include "xenia/ui/windowed_app_context_ios.h"
#include "xenia/vfs/devices/xcontent_container_device.h"
#include "xenia/vfs/iso_metadata.h"
#include "xenia/vfs/xex_metadata.h"
#include "xenia/xbox.h"

DECLARE_path(log_file);

// Forward declarations of the Objective-C classes.
@class XeniaAppDelegate;
@class XeniaViewController;
@class XeniaMetalView;

extern "C" int csops(pid_t pid, unsigned int ops, void* useraddr, size_t usersize);
extern "C" int ptrace(int request, pid_t pid, caddr_t addr, int data);

#ifndef CS_OPS_STATUS
#define CS_OPS_STATUS 0
#endif
#ifndef CS_DEBUGGED
#define CS_DEBUGGED 0x10000000
#endif
#ifndef PT_TRACE_ME
#define PT_TRACE_ME 0
#endif

static BOOL xe_is_cs_debugged(void) {
  int flags = 0;
  return !csops(getpid(), CS_OPS_STATUS, &flags, sizeof(flags)) && (flags & CS_DEBUGGED);
}

static BOOL xe_can_mmap_exec_page(void) {
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) page_size = 16384;
  void* test =
      mmap(NULL, (size_t)page_size, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (test == MAP_FAILED) {
    return NO;
  }
  munmap(test, (size_t)page_size);
  return YES;
}

static BOOL xe_enable_ptrace_jit_hack(void) {
#if TARGET_OS_TV
  return NO;
#else
  if (xe_is_cs_debugged()) {
    return YES;
  }

  // Keep this minimal: only request traced state to obtain CS_DEBUGGED.
  // Do not alter exception ports; Xenia depends on SIGBUS/SIGSEGV handling.
  if (ptrace(PT_TRACE_ME, 0, NULL, 0) < 0) {
    return NO;
  }
  return YES;
#endif
}

// ---------------------------------------------------------------------------
// JIT availability check -- tests whether executable memory can be mapped.
// Requires CS_DEBUGGED (set by StikDebug, AltJIT, SideJITServer, etc.).
// ---------------------------------------------------------------------------
static BOOL xe_check_jit_available(void) { return xe_is_cs_debugged() && xe_can_mmap_exec_page(); }

static std::filesystem::path xe_get_ios_documents_path() {
  @autoreleasepool {
    NSArray* paths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    if (paths.count > 0) {
      return std::filesystem::path([paths[0] UTF8String]);
    }
    return std::filesystem::path([NSHomeDirectory() UTF8String]) / "Documents";
  }
}

static void xe_request_orientation(UIViewController* view_controller,
                                   UIInterfaceOrientationMask mask,
                                   UIInterfaceOrientation orientation) {
  if (!view_controller) {
    return;
  }
#if !TARGET_OS_TV
  [view_controller setNeedsUpdateOfSupportedInterfaceOrientations];
  if (@available(iOS 16.0, *)) {
    UIWindowScene* scene = view_controller.view.window.windowScene;
    if (!scene) {
      for (UIScene* connected_scene in [UIApplication sharedApplication].connectedScenes) {
        if ([connected_scene isKindOfClass:[UIWindowScene class]]) {
          scene = (UIWindowScene*)connected_scene;
          break;
        }
      }
    }
    if (scene) {
      UIWindowSceneGeometryPreferencesIOS* preferences =
          [[UIWindowSceneGeometryPreferencesIOS alloc] initWithInterfaceOrientations:mask];
      [scene requestGeometryUpdateWithPreferences:preferences
                                     errorHandler:^(NSError* error) {
                                       XELOGW("iOS: Orientation geometry update failed: {}",
                                              [[error localizedDescription] UTF8String]);
                                     }];
    }
  }
  [[UIDevice currentDevice] setValue:@(orientation) forKey:@"orientation"];
  [UIViewController attemptRotationToDeviceOrientation];
#endif
}

static void xe_request_landscape_orientation(UIViewController* view_controller) {
  xe_request_orientation(view_controller, UIInterfaceOrientationMaskLandscape,
                         UIInterfaceOrientationLandscapeRight);
}

static void xe_request_portrait_orientation(UIViewController* view_controller) {
  xe_request_orientation(view_controller, UIInterfaceOrientationMaskPortrait,
                         UIInterfaceOrientationPortrait);
}

// ---------------------------------------------------------------------------
// XeniaMetalView - a UIView backed by a CAMetalLayer.
// ---------------------------------------------------------------------------
@interface XeniaMetalView : UIView
@end

@implementation XeniaMetalView

+ (Class)layerClass {
  return [CAMetalLayer class];
}

@end

namespace {

enum class IOSConfigControlType {
  kToggle,
  kChoiceInt32,
  kChoiceUInt64,
  kAction,
};

enum class IOSConfigAction {
  kNone,
  kViewRecentLog,
};

struct IOSConfigChoice {
  std::string title;
  int64_t value = 0;
};

struct IOSConfigItem {
  std::string key;
  std::string title;
  std::string subtitle;
  IOSConfigControlType control_type = IOSConfigControlType::kToggle;
  bool bool_value = false;
  int64_t choice_value = 0;
  IOSConfigAction action = IOSConfigAction::kNone;
  std::vector<IOSConfigChoice> choices;
};

struct IOSConfigSection {
  std::string title;
  std::string footer;
  std::vector<IOSConfigItem> items;
};

std::string TrimAscii(std::string value) {
  size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
    ++start;
  }
  size_t end = value.size();
  while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(start, end - start);
}

NSString* ToNSString(const std::string& value) {
  return [NSString stringWithUTF8String:value.c_str()];
}

cvar::IConfigVar* GetConfigVar(const std::string& key) {
  if (!cvar::ConfigVars) {
    return nullptr;
  }
  auto it = cvar::ConfigVars->find(key);
  if (it == cvar::ConfigVars->end()) {
    return nullptr;
  }
  return it->second;
}

bool HasConfigVar(const std::string& key) { return GetConfigVar(key) != nullptr; }

std::string GetConfigVarString(const std::string& key, const std::string& fallback) {
  cvar::IConfigVar* var = GetConfigVar(key);
  if (!var) {
    return fallback;
  }
  return TrimAscii(var->config_value());
}

bool ParseBoolString(const std::string& text, bool* value_out) {
  if (!value_out) {
    return false;
  }
  std::string lower = text;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (lower == "true" || lower == "1") {
    *value_out = true;
    return true;
  }
  if (lower == "false" || lower == "0") {
    *value_out = false;
    return true;
  }
  return false;
}

bool ParseInt64String(const std::string& text, int64_t* value_out) {
  if (!value_out) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  long long parsed = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || !end || *end != '\0') {
    return false;
  }
  *value_out = static_cast<int64_t>(parsed);
  return true;
}

bool SetConfigVarBool(const std::string& key, bool value) {
  cvar::IConfigVar* var = GetConfigVar(key);
  if (!var) {
    XELOGW("iOS settings: missing config var '{}'", key);
    return false;
  }
  toml::value node(value);
  var->LoadConfigValue(&node);
  return true;
}

bool SetConfigVarInt32(const std::string& key, int32_t value) {
  cvar::IConfigVar* var = GetConfigVar(key);
  if (!var) {
    XELOGW("iOS settings: missing config var '{}'", key);
    return false;
  }
  toml::value node(value);
  var->LoadConfigValue(&node);
  return true;
}

bool SetConfigVarUInt64(const std::string& key, uint64_t value) {
  cvar::IConfigVar* var = GetConfigVar(key);
  if (!var) {
    XELOGW("iOS settings: missing config var '{}'", key);
    return false;
  }
  toml::value node(value);
  var->LoadConfigValue(&node);
  return true;
}

void AddBoolSetting(std::vector<IOSConfigItem>& items, const std::string& key,
                    const std::string& title, const std::string& subtitle, bool fallback) {
  if (!HasConfigVar(key)) {
    return;
  }
  IOSConfigItem item;
  item.key = key;
  item.title = title;
  item.subtitle = subtitle;
  item.control_type = IOSConfigControlType::kToggle;
  item.bool_value = fallback;
  ParseBoolString(GetConfigVarString(key, fallback ? "true" : "false"), &item.bool_value);
  items.push_back(std::move(item));
}

void AddChoiceSetting(std::vector<IOSConfigItem>& items, IOSConfigControlType control_type,
                      const std::string& key, const std::string& title, const std::string& subtitle,
                      int64_t fallback, std::vector<IOSConfigChoice> choices) {
  if (!HasConfigVar(key) || choices.empty()) {
    return;
  }
  IOSConfigItem item;
  item.key = key;
  item.title = title;
  item.subtitle = subtitle;
  item.control_type = control_type;
  item.choice_value = fallback;
  ParseInt64String(GetConfigVarString(key, std::to_string(fallback)), &item.choice_value);
  item.choices = std::move(choices);
  bool found = false;
  for (const IOSConfigChoice& choice : item.choices) {
    if (choice.value == item.choice_value) {
      found = true;
      break;
    }
  }
  if (!found) {
    item.choice_value = item.choices.front().value;
  }
  items.push_back(std::move(item));
}

void AddActionSetting(std::vector<IOSConfigItem>& items, IOSConfigAction action,
                      const std::string& title, const std::string& subtitle) {
  IOSConfigItem item;
  item.title = title;
  item.subtitle = subtitle;
  item.control_type = IOSConfigControlType::kAction;
  item.action = action;
  items.push_back(std::move(item));
}

std::string ChoiceTitleForItem(const IOSConfigItem& item) {
  for (const IOSConfigChoice& choice : item.choices) {
    if (choice.value == item.choice_value) {
      return choice.title;
    }
  }
  return item.choices.empty() ? std::string() : item.choices.front().title;
}

std::filesystem::path GetLogFilePath() {
  if (!cvars::log_file.empty()) {
    return cvars::log_file;
  }
  return xe_get_ios_documents_path() / "xenia.log";
}

std::string ReadFileTail(const std::filesystem::path& path, size_t max_bytes) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::string();
  }

  file.seekg(0, std::ios::end);
  const std::streamoff end = file.tellg();
  if (end <= 0) {
    return std::string();
  }

  const std::streamoff max_read = static_cast<std::streamoff>(max_bytes);
  const std::streamoff start = end > max_read ? (end - max_read) : 0;
  file.seekg(start, std::ios::beg);

  std::string content(static_cast<size_t>(end - start), '\0');
  file.read(content.data(), static_cast<std::streamsize>(content.size()));
  content.resize(static_cast<size_t>(file.gcount()));

  if (start > 0) {
    size_t first_newline = content.find('\n');
    if (first_newline != std::string::npos) {
      content.erase(0, first_newline + 1);
    }
  }
  return content;
}

NSString* DecodeLogBytesToNSString(const std::string& content) {
  if (content.empty()) {
    return @"";
  }

  NSData* log_data = [NSData dataWithBytes:content.data() length:content.size()];
  NSString* decoded = [[NSString alloc] initWithData:log_data encoding:NSUTF8StringEncoding];
  if (!decoded) {
    decoded = [[NSString alloc] initWithData:log_data
                                    encoding:CFStringConvertEncodingToNSStringEncoding(
                                                 kCFStringEncodingWindowsLatin1)];
  }
  if (!decoded) {
    decoded = [[NSString alloc] initWithData:log_data encoding:NSISOLatin1StringEncoding];
  }
  if (decoded) {
    return decoded;
  }

  // Last-resort byte-preserving fallback so the UI never shows a hard decode
  // failure marker.
  NSMutableString* fallback = [NSMutableString stringWithCapacity:content.size()];
  for (unsigned char byte : content) {
    if (byte == '\n' || byte == '\r' || byte == '\t' ||
        (byte >= 0x20 && byte <= 0x7E)) {
      [fallback appendFormat:@"%c", byte];
    } else {
      [fallback appendFormat:@"\\x%02X", byte];
    }
  }
  return fallback;
}

struct IOSDiscoveredGame {
  std::filesystem::path path;
  std::string title;
  uint32_t title_id = 0;
  std::vector<uint8_t> icon_data;
};

std::string ToLowerAsciiCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

bool IsISOPath(const std::filesystem::path& path) {
  return ToLowerAsciiCopy(path.extension().string()) == ".iso";
}

bool IsDefaultXexPath(const std::filesystem::path& path) {
  return ToLowerAsciiCopy(path.filename().string()) == "default.xex";
}

bool IsLikelyGodPath(const std::filesystem::path& path) {
  if (!path.has_filename()) {
    return false;
  }
  std::filesystem::path parent = path.parent_path();
  while (!parent.empty()) {
    std::string name_lower = ToLowerAsciiCopy(parent.filename().string());
    if (name_lower == "00007000" || name_lower == "00004000") {
      return true;
    }
    std::filesystem::path next = parent.parent_path();
    if (next == parent) {
      break;
    }
    parent = next;
  }
  return false;
}

std::string FormatTitleID(uint32_t title_id) {
  if (!title_id) {
    return std::string();
  }
  char buffer[9] = {};
  std::snprintf(buffer, sizeof(buffer), "%08X", title_id);
  return std::string(buffer);
}

std::string DisplayNameFromXexMetadata(const std::filesystem::path& path,
                                       const std::optional<xe::vfs::XexMetadata>& metadata) {
  if (!metadata.has_value()) {
    return path.stem().string();
  }
  if (!metadata->module_name.empty()) {
    return metadata->module_name;
  }
  std::string title_id = FormatTitleID(metadata->title_id);
  if (!title_id.empty()) {
    return "Title " + title_id;
  }
  return path.stem().string();
}

bool BuildDiscoveredGameFromPath(const std::filesystem::path& path, IOSDiscoveredGame* game_out) {
  if (!game_out) {
    return false;
  }

  IOSDiscoveredGame game;
  game.path = path;
  if (IsISOPath(path)) {
    auto metadata = xe::vfs::ExtractIsoMetadata(path);
    if (metadata.has_value()) {
      game.title_id = metadata->title_id;
    }
    game.title = DisplayNameFromXexMetadata(path, metadata);
    *game_out = std::move(game);
    return true;
  }

  if (IsDefaultXexPath(path)) {
    auto metadata = xe::vfs::ExtractXexMetadata(path);
    if (metadata.has_value()) {
      game.title_id = metadata->title_id;
    }
    game.title = DisplayNameFromXexMetadata(path, metadata);
    *game_out = std::move(game);
    return true;
  }

  if (!IsLikelyGodPath(path)) {
    return false;
  }

  auto header = xe::vfs::XContentContainerDevice::ReadContainerHeader(path);
  if (!header || !header->content_header.is_magic_valid()) {
    return false;
  }

  xe::XContentType content_type =
      static_cast<xe::XContentType>(header->content_metadata.content_type.get());
  if (content_type != xe::XContentType::kXbox360Title &&
      content_type != xe::XContentType::kInstalledGame) {
    return false;
  }

  game.title_id = header->content_metadata.execution_info.title_id;
  std::string display_name =
      xe::to_utf8(header->content_metadata.display_name(xe::XLanguage::kEnglish));
  if (display_name.empty()) {
    display_name = xe::to_utf8(header->content_metadata.title_name());
  }
  if (display_name.empty()) {
    std::string title_id = FormatTitleID(game.title_id);
    game.title = title_id.empty() ? path.stem().string() : "Title " + title_id;
  } else {
    game.title = display_name;
  }

  uint32_t thumb_size = header->content_metadata.title_thumbnail_size;
  if (thumb_size > 0 && thumb_size <= xe::vfs::XContentMetadata::kThumbLengthV1) {
    game.icon_data.assign(header->content_metadata.title_thumbnail,
                          header->content_metadata.title_thumbnail + thumb_size);
  }

  *game_out = std::move(game);
  return true;
}

std::vector<IOSConfigSection> BuildIOSConfigSections() {
  std::vector<IOSConfigSection> sections;

  IOSConfigSection graphics;
  graphics.title = "Graphics";
  graphics.footer = "These options affect image scaling and presentation.";
  AddBoolSetting(graphics.items, "present_letterbox", "Keep Aspect Ratio",
                 "Adds letterboxing to preserve image proportions.", true);
  AddChoiceSetting(graphics.items, IOSConfigControlType::kChoiceInt32, "present_safe_area_x",
                   "Safe Area (Horizontal)", "How much width is kept before cropping.", 100,
                   {{"90%", 90}, {"95%", 95}, {"100%", 100}});
  AddChoiceSetting(graphics.items, IOSConfigControlType::kChoiceInt32, "present_safe_area_y",
                   "Safe Area (Vertical)", "How much height is kept before cropping.", 100,
                   {{"90%", 90}, {"95%", 95}, {"100%", 100}});
  AddBoolSetting(graphics.items, "metal_presenter_use_backing_scale", "Retina Backing Scale",
                 "When off, render at logical size to reduce GPU load.", false);
  AddChoiceSetting(graphics.items, IOSConfigControlType::kChoiceInt32, "anisotropic_override",
                   "Anisotropic Filtering", "Texture filtering override level.", -1,
                   {{"Auto", -1}, {"Off", 0}, {"2x", 2}, {"4x", 4}, {"8x", 8}, {"16x", 16}});
  if (!graphics.items.empty()) {
    sections.push_back(std::move(graphics));
  }

  IOSConfigSection performance;
  performance.title = "Performance";
  performance.footer = "Some changes may require relaunching the current game.";
  AddChoiceSetting(performance.items, IOSConfigControlType::kChoiceUInt64, "framerate_limit",
                   "Frame Rate Limit", "Host presentation FPS cap. Guest timing is separate.", 0,
                   {{"Unlimited", 0}, {"30 FPS", 30}, {"60 FPS", 60}, {"120 FPS", 120}});
  AddBoolSetting(performance.items, "guest_display_refresh_cap", "Cap Guest Display Refresh",
                 "Runs guest vblank at PAL/NTSC rate instead of unlimited.", true);
  AddBoolSetting(performance.items, "async_shader_compilation", "Async Shader Compilation",
                 "Backend-dependent. Helps where supported; Metal support is currently limited.",
                 true);
  AddBoolSetting(performance.items, "half_pixel_offset", "Half-Pixel Offset",
                 "Compatibility adjustment for D3D9-style sampling.", true);
  AddBoolSetting(performance.items, "occlusion_query_enable", "Hardware Occlusion Queries",
                 "More accurate but can reduce performance.", false);
  if (!performance.items.empty()) {
    sections.push_back(std::move(performance));
  }

  IOSConfigSection compatibility;
  compatibility.title = "Compatibility";
  compatibility.footer = "Saved to xenia-edge.config.toml in the app Documents folder.";
  AddBoolSetting(compatibility.items, "gpu_allow_invalid_fetch_constants",
                 "Allow Invalid Fetch Constants",
                 "Unsafe workaround for games with bad fetch constants.", true);
  AddBoolSetting(compatibility.items, "gpu_allow_invalid_upload_range",
                 "Allow Invalid Upload Range", "Allows reads from pages marked no-access.", false);
  AddBoolSetting(compatibility.items, "ios_jit_brk_use_universal_0xf00d",
                 "Universal JIT Breakpoint",
                 "Use brk #0xF00D (x16 command dispatch) instead of legacy "
                 "brk #0x69.",
                 false);
  if (!compatibility.items.empty()) {
    sections.push_back(std::move(compatibility));
  }

  IOSConfigSection diagnostics;
  diagnostics.title = "Diagnostics";
  diagnostics.footer = "Logs are stored in Documents/xenia.log and can be viewed without Xcode.";
  AddChoiceSetting(diagnostics.items, IOSConfigControlType::kChoiceInt32, "log_level",
                   "Log Verbosity", "Lower values reduce log noise and file size.", 2,
                   {{"Errors Only", 0}, {"Warnings", 1}, {"Info", 2}, {"Debug", 3}});
  AddActionSetting(diagnostics.items, IOSConfigAction::kViewRecentLog, "View Live Log",
                   "Open a live-updating xenia.log viewer.");
  if (!diagnostics.items.empty()) {
    sections.push_back(std::move(diagnostics));
  }

  return sections;
}

bool ApplyIOSConfigSections(const std::vector<IOSConfigSection>& sections) {
  bool ok = true;
  for (const IOSConfigSection& section : sections) {
    for (const IOSConfigItem& item : section.items) {
      switch (item.control_type) {
        case IOSConfigControlType::kToggle:
          ok &= SetConfigVarBool(item.key, item.bool_value);
          break;
        case IOSConfigControlType::kChoiceInt32:
          ok &= SetConfigVarInt32(item.key, static_cast<int32_t>(item.choice_value));
          break;
        case IOSConfigControlType::kChoiceUInt64:
          ok &= SetConfigVarUInt64(item.key, static_cast<uint64_t>(item.choice_value));
          break;
        case IOSConfigControlType::kAction:
          break;
      }
    }
  }
  config::SaveConfig();
  return ok;
}

}  // namespace

@interface XeniaConfigViewController : UITableViewController
@end

typedef void (^IOSChoiceSelectionHandler)(int64_t value);

@interface XeniaChoiceListViewController : UITableViewController
- (instancetype)initWithTitle:(NSString*)title
                     subtitle:(NSString*)subtitle
                      choices:(const std::vector<IOSConfigChoice>&)choices
                selectedValue:(int64_t)selectedValue
                  onSelection:(IOSChoiceSelectionHandler)onSelection;
@end

@interface XeniaLogViewController : UIViewController
@end

@interface XeniaLandscapeNavigationController : UINavigationController
@end

@interface XeniaGameTileCell : UICollectionViewCell
@property(nonatomic, strong) UIImageView* iconView;
@property(nonatomic, strong) UILabel* titleLabel;
@end

@implementation XeniaGameTileCell

- (instancetype)initWithFrame:(CGRect)frame {
  self = [super initWithFrame:frame];
  if (!self) {
    return nil;
  }

  self.contentView.backgroundColor = [UIColor colorWithWhite:1.0 alpha:0.08];
  self.contentView.layer.cornerRadius = 14;
  self.contentView.layer.borderWidth = 1.0;
  self.contentView.layer.borderColor = [UIColor colorWithWhite:1.0 alpha:0.08].CGColor;

  self.iconView = [[UIImageView alloc] init];
  self.iconView.translatesAutoresizingMaskIntoConstraints = NO;
  self.iconView.contentMode = UIViewContentModeScaleAspectFit;
  self.iconView.layer.cornerRadius = 12;
  self.iconView.clipsToBounds = YES;
  self.iconView.backgroundColor = [UIColor colorWithWhite:1.0 alpha:0.12];
  [self.contentView addSubview:self.iconView];

  self.titleLabel = [[UILabel alloc] init];
  self.titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
  self.titleLabel.font = [UIFont systemFontOfSize:12 weight:UIFontWeightSemibold];
  self.titleLabel.textColor = [UIColor whiteColor];
  self.titleLabel.textAlignment = NSTextAlignmentCenter;
  self.titleLabel.numberOfLines = 2;
  [self.contentView addSubview:self.titleLabel];

  [NSLayoutConstraint activateConstraints:@[
    [self.iconView.topAnchor constraintEqualToAnchor:self.contentView.topAnchor constant:10],
    [self.iconView.leadingAnchor constraintEqualToAnchor:self.contentView.leadingAnchor
                                                constant:10],
    [self.iconView.trailingAnchor constraintEqualToAnchor:self.contentView.trailingAnchor
                                                 constant:-10],
    [self.iconView.heightAnchor constraintEqualToAnchor:self.iconView.widthAnchor],
    [self.titleLabel.topAnchor constraintEqualToAnchor:self.iconView.bottomAnchor constant:6],
    [self.titleLabel.leadingAnchor constraintEqualToAnchor:self.contentView.leadingAnchor
                                                  constant:6],
    [self.titleLabel.trailingAnchor constraintEqualToAnchor:self.contentView.trailingAnchor
                                                   constant:-6],
    [self.titleLabel.bottomAnchor constraintLessThanOrEqualToAnchor:self.contentView.bottomAnchor
                                                           constant:-8],
  ]];

  return self;
}

@end

@implementation XeniaChoiceListViewController {
  std::vector<IOSConfigChoice> choices_;
  int64_t selected_value_;
  NSString* subtitle_;
  IOSChoiceSelectionHandler on_selection_;
}

- (instancetype)initWithTitle:(NSString*)title
                     subtitle:(NSString*)subtitle
                      choices:(const std::vector<IOSConfigChoice>&)choices
                selectedValue:(int64_t)selectedValue
                  onSelection:(IOSChoiceSelectionHandler)onSelection {
  self = [super initWithStyle:UITableViewStyleInsetGrouped];
  if (self) {
    self.title = title;
    subtitle_ = [subtitle copy];
    choices_ = choices;
    selected_value_ = selectedValue;
    on_selection_ = [onSelection copy];
  }
  return self;
}

- (void)viewDidLoad {
  [super viewDidLoad];
  self.tableView.backgroundColor = [UIColor systemBackgroundColor];
  self.tableView.separatorInset = UIEdgeInsetsMake(0, 16, 0, 16);

  if (subtitle_.length > 0) {
    UILabel* header_label = [[UILabel alloc] initWithFrame:CGRectZero];
    header_label.text = subtitle_;
    header_label.textColor = [UIColor secondaryLabelColor];
    header_label.font = [UIFont systemFontOfSize:13];
    header_label.numberOfLines = 0;
    header_label.textAlignment = NSTextAlignmentLeft;
    header_label.translatesAutoresizingMaskIntoConstraints = NO;

    UIView* header = [[UIView alloc] initWithFrame:CGRectMake(0, 0, 1, 56)];
    [header addSubview:header_label];
    [NSLayoutConstraint activateConstraints:@[
      [header_label.leadingAnchor constraintEqualToAnchor:header.leadingAnchor constant:18],
      [header_label.trailingAnchor constraintEqualToAnchor:header.trailingAnchor constant:-18],
      [header_label.topAnchor constraintEqualToAnchor:header.topAnchor constant:8],
      [header_label.bottomAnchor constraintEqualToAnchor:header.bottomAnchor constant:-8],
    ]];
    self.tableView.tableHeaderView = header;
  }
}

- (NSInteger)tableView:(UITableView* __unused)tableView
    numberOfRowsInSection:(NSInteger)__unused section {
  return static_cast<NSInteger>(choices_.size());
}

- (UITableViewCell*)tableView:(UITableView*)tableView
        cellForRowAtIndexPath:(NSIndexPath*)indexPath {
  static NSString* const kChoiceCellIdentifier = @"XeniaChoiceCell";
  UITableViewCell* cell = [tableView dequeueReusableCellWithIdentifier:kChoiceCellIdentifier];
  if (!cell) {
    cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault
                                  reuseIdentifier:kChoiceCellIdentifier];
  }
  if (indexPath.row < 0 || indexPath.row >= static_cast<NSInteger>(choices_.size())) {
    cell.textLabel.text = @"";
    cell.accessoryType = UITableViewCellAccessoryNone;
    return cell;
  }
  const IOSConfigChoice& choice = choices_[indexPath.row];
  cell.textLabel.text = ToNSString(choice.title);
  cell.accessoryType = (choice.value == selected_value_) ? UITableViewCellAccessoryCheckmark
                                                         : UITableViewCellAccessoryNone;
  return cell;
}

- (void)tableView:(UITableView*)tableView didSelectRowAtIndexPath:(NSIndexPath*)indexPath {
  if (indexPath.row < 0 || indexPath.row >= static_cast<NSInteger>(choices_.size())) {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    return;
  }
  const IOSConfigChoice& choice = choices_[indexPath.row];
  selected_value_ = choice.value;
  if (on_selection_) {
    on_selection_(selected_value_);
  }
  [self.navigationController popViewControllerAnimated:YES];
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
  return UIInterfaceOrientationMaskAllButUpsideDown;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationPortrait;
}

@end

@implementation XeniaLogViewController {
  UITextView* textView_;
  UILabel* footerLabel_;
  NSTimer* autoRefreshTimer_;
}

- (void)viewDidLoad {
  [super viewDidLoad];
  self.title = @"Live Log";
  self.view.backgroundColor = [UIColor systemBackgroundColor];

  UIBarButtonItem* refresh_button =
      [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemRefresh
                                                    target:self
                                                    action:@selector(reloadLogTapped:)];
  UIBarButtonItem* share_button =
      [[UIBarButtonItem alloc] initWithTitle:@"Share"
                                       style:UIBarButtonItemStylePlain
                                      target:self
                                      action:@selector(shareLogTapped:)];
  self.navigationItem.rightBarButtonItems = @[ refresh_button, share_button ];

  textView_ = [[UITextView alloc] init];
  textView_.translatesAutoresizingMaskIntoConstraints = NO;
  textView_.editable = NO;
  textView_.selectable = YES;
  textView_.alwaysBounceVertical = YES;
  textView_.backgroundColor = [UIColor secondarySystemBackgroundColor];
  textView_.font = [UIFont monospacedSystemFontOfSize:12 weight:UIFontWeightRegular];
  textView_.textContainerInset = UIEdgeInsetsMake(12, 12, 12, 12);
  [self.view addSubview:textView_];

  footerLabel_ = [[UILabel alloc] init];
  footerLabel_.translatesAutoresizingMaskIntoConstraints = NO;
  footerLabel_.textColor = [UIColor secondaryLabelColor];
  footerLabel_.font = [UIFont systemFontOfSize:12];
  footerLabel_.numberOfLines = 2;
  [self.view addSubview:footerLabel_];

  UILayoutGuide* guide = self.view.safeAreaLayoutGuide;
  [NSLayoutConstraint activateConstraints:@[
    [textView_.topAnchor constraintEqualToAnchor:guide.topAnchor constant:8],
    [textView_.leadingAnchor constraintEqualToAnchor:guide.leadingAnchor constant:10],
    [textView_.trailingAnchor constraintEqualToAnchor:guide.trailingAnchor constant:-10],
    [textView_.bottomAnchor constraintEqualToAnchor:footerLabel_.topAnchor constant:-8],
    [footerLabel_.leadingAnchor constraintEqualToAnchor:guide.leadingAnchor constant:12],
    [footerLabel_.trailingAnchor constraintEqualToAnchor:guide.trailingAnchor constant:-12],
    [footerLabel_.bottomAnchor constraintEqualToAnchor:guide.bottomAnchor constant:-8],
  ]];
}

- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];
  [self updateCloseButton];
  [self reloadLog];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
  [self startAutoRefresh];
}

- (void)viewWillDisappear:(BOOL)animated {
  [super viewWillDisappear:animated];
  [self stopAutoRefresh];
}

- (void)dealloc {
  [self stopAutoRefresh];
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
  return UIInterfaceOrientationMaskAllButUpsideDown;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationPortrait;
}

- (void)reloadLogTapped:(id)sender {
  [self reloadLog];
}

- (void)closeTapped:(id)sender {
  [self dismissViewControllerAnimated:YES completion:nil];
}

- (BOOL)shouldShowCloseButton {
  UINavigationController* nav = self.navigationController;
  if (!nav || nav.presentingViewController == nil) {
    return NO;
  }
  return nav.viewControllers.count > 0 && nav.viewControllers.firstObject == self;
}

- (void)updateCloseButton {
  if ([self shouldShowCloseButton]) {
    self.navigationItem.leftBarButtonItem =
        [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemDone
                                                      target:self
                                                      action:@selector(closeTapped:)];
  } else {
    self.navigationItem.leftBarButtonItem = nil;
  }
}

- (void)startAutoRefresh {
  if (autoRefreshTimer_) {
    return;
  }
  autoRefreshTimer_ = [NSTimer scheduledTimerWithTimeInterval:0.5
                                                        target:self
                                                      selector:@selector(autoRefreshTick:)
                                                      userInfo:nil
                                                       repeats:YES];
  autoRefreshTimer_.tolerance = 0.15;
}

- (void)stopAutoRefresh {
  if (!autoRefreshTimer_) {
    return;
  }
  [autoRefreshTimer_ invalidate];
  autoRefreshTimer_ = nil;
}

- (void)autoRefreshTick:(NSTimer* __unused)timer {
  [self reloadLog];
}

- (BOOL)isNearBottom {
  CGFloat content_height = textView_.contentSize.height;
  CGFloat visible_height = CGRectGetHeight(textView_.bounds);
  if (content_height <= visible_height + 1.0) {
    return YES;
  }
  CGFloat bottom = textView_.contentOffset.y + visible_height;
  return bottom >= content_height - 36.0;
}

- (void)shareLogTapped:(id)sender {
  std::filesystem::path log_path = GetLogFilePath();
  NSString* log_path_ns = ToNSString(log_path.string());
  if (![[NSFileManager defaultManager] fileExistsAtPath:log_path_ns]) {
    UIAlertController* alert =
        [UIAlertController alertControllerWithTitle:@"Log Not Found"
                                            message:@"No xenia.log file exists yet."
                                     preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"OK"
                                              style:UIAlertActionStyleDefault
                                            handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
    return;
  }

  NSURL* log_url = [NSURL fileURLWithPath:log_path_ns];
  UIActivityViewController* share_controller =
      [[UIActivityViewController alloc] initWithActivityItems:@[ log_url ]
                                        applicationActivities:nil];
  UIPopoverPresentationController* popover = share_controller.popoverPresentationController;
  if (popover) {
    popover.barButtonItem = self.navigationItem.rightBarButtonItems.lastObject;
  }
  [self presentViewController:share_controller animated:YES completion:nil];
}

- (void)reloadLog {
  static constexpr size_t kMaxLogBytes = 256 * 1024;
  BOOL should_follow_tail = [self isNearBottom];
  std::filesystem::path log_path = GetLogFilePath();
  std::string content = ReadFileTail(log_path, kMaxLogBytes);
  NSString* log_path_ns = ToNSString(log_path.string());

  NSString* display_text = nil;
  if (content.empty()) {
    display_text =
        [NSString stringWithFormat:@"No recent log data found.\n\nExpected file:\n%@\n\n"
                                   @"Launch a game and keep this open. This view updates "
                                   @"automatically.",
                                   log_path_ns];
  } else {
    display_text = DecodeLogBytesToNSString(content);
  }

  if (!display_text) {
    display_text = @"";
  }
  if (![textView_.text isEqualToString:display_text]) {
    textView_.text = display_text;
  }
  if (should_follow_tail) {
    [textView_ scrollRangeToVisible:NSMakeRange(textView_.text.length, 0)];
  }

  footerLabel_.text =
      [NSString stringWithFormat:@"Live tail (0.5s). Showing last %zu KB from %@",
                                 kMaxLogBytes / 1024, log_path_ns];
}

@end

@implementation XeniaLandscapeNavigationController

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
  return UIInterfaceOrientationMaskAllButUpsideDown;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationPortrait;
}

- (BOOL)shouldAutorotate {
  return YES;
}

@end

@implementation XeniaConfigViewController {
  std::vector<IOSConfigSection> sections_;
  BOOL hasPendingChanges_;
  UIBarButtonItem* saveButton_;
}

- (void)viewDidLoad {
  [super viewDidLoad];
  self.title = @"Settings";
  self.tableView.backgroundColor = [UIColor systemBackgroundColor];
  self.tableView.separatorInset = UIEdgeInsetsMake(0, 16, 0, 16);
  sections_ = BuildIOSConfigSections();
  hasPendingChanges_ = NO;

  self.navigationItem.leftBarButtonItem =
      [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemCancel
                                                    target:self
                                                    action:@selector(cancelTapped:)];
  saveButton_ = [[UIBarButtonItem alloc] initWithTitle:@"Save"
                                                 style:UIBarButtonItemStyleDone
                                                target:self
                                                action:@selector(saveTapped:)];
  saveButton_.enabled = NO;
  self.navigationItem.rightBarButtonItem = saveButton_;
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
  return UIInterfaceOrientationMaskAllButUpsideDown;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationPortrait;
}

- (void)markPendingChanges {
  hasPendingChanges_ = YES;
  saveButton_.enabled = YES;
}

- (IOSConfigItem*)itemAtIndexPath:(NSIndexPath*)indexPath {
  if (indexPath.section < 0 || indexPath.section >= static_cast<NSInteger>(sections_.size())) {
    return nullptr;
  }
  IOSConfigSection& section = sections_[indexPath.section];
  if (indexPath.row < 0 || indexPath.row >= static_cast<NSInteger>(section.items.size())) {
    return nullptr;
  }
  return &section.items[indexPath.row];
}

- (NSInteger)numberOfSectionsInTableView:(UITableView*)tableView {
  return static_cast<NSInteger>(sections_.size());
}

- (NSInteger)tableView:(UITableView*)tableView numberOfRowsInSection:(NSInteger)section {
  if (section < 0 || section >= static_cast<NSInteger>(sections_.size())) {
    return 0;
  }
  return static_cast<NSInteger>(sections_[section].items.size());
}

- (NSString*)tableView:(UITableView*)tableView titleForHeaderInSection:(NSInteger)section {
  if (section < 0 || section >= static_cast<NSInteger>(sections_.size())) {
    return nil;
  }
  return ToNSString(sections_[section].title);
}

- (NSString*)tableView:(UITableView*)tableView titleForFooterInSection:(NSInteger)section {
  if (section < 0 || section >= static_cast<NSInteger>(sections_.size())) {
    return nil;
  }
  return ToNSString(sections_[section].footer);
}

- (UITableViewCell*)tableView:(UITableView*)tableView
        cellForRowAtIndexPath:(NSIndexPath*)indexPath {
  static NSString* const kCellIdentifier = @"XeniaConfigCell";
  UITableViewCell* cell = [tableView dequeueReusableCellWithIdentifier:kCellIdentifier];
  if (!cell) {
    cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleSubtitle
                                  reuseIdentifier:kCellIdentifier];
  }

  IOSConfigItem* item = [self itemAtIndexPath:indexPath];
  if (!item) {
    cell.textLabel.text = @"";
    cell.detailTextLabel.text = @"";
    cell.accessoryType = UITableViewCellAccessoryNone;
    cell.accessoryView = nil;
    cell.selectionStyle = UITableViewCellSelectionStyleNone;
    return cell;
  }

  cell.textLabel.text = ToNSString(item->title);
  cell.detailTextLabel.textColor = [UIColor secondaryLabelColor];
  cell.detailTextLabel.numberOfLines = 2;
  cell.textLabel.numberOfLines = 1;

  if (item->control_type == IOSConfigControlType::kToggle) {
    cell.textLabel.textColor = [UIColor labelColor];
    cell.detailTextLabel.text = ToNSString(item->subtitle);
    UISwitch* toggle = [[UISwitch alloc] init];
    toggle.on = item->bool_value;
    [toggle addTarget:self
                  action:@selector(toggleChanged:)
        forControlEvents:UIControlEventValueChanged];
    cell.accessoryView = toggle;
    cell.accessoryType = UITableViewCellAccessoryNone;
    cell.selectionStyle = UITableViewCellSelectionStyleNone;
  } else if (item->control_type == IOSConfigControlType::kAction) {
    cell.textLabel.textColor = self.view.tintColor;
    cell.detailTextLabel.text = ToNSString(item->subtitle);
    cell.accessoryView = nil;
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
    cell.selectionStyle = UITableViewCellSelectionStyleDefault;
  } else {
    cell.textLabel.textColor = [UIColor labelColor];
    std::string value_title = ChoiceTitleForItem(*item);
    std::string subtitle = item->subtitle;
    if (!value_title.empty()) {
      cell.detailTextLabel.text = ToNSString(value_title + " · " + subtitle);
    } else {
      cell.detailTextLabel.text = ToNSString(subtitle);
    }
    cell.accessoryView = nil;
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
    cell.selectionStyle = UITableViewCellSelectionStyleDefault;
  }

  return cell;
}

- (void)toggleChanged:(UISwitch*)sender {
  CGPoint point = [sender convertPoint:CGPointZero toView:self.tableView];
  NSIndexPath* indexPath = [self.tableView indexPathForRowAtPoint:point];
  if (!indexPath) {
    return;
  }
  IOSConfigItem* item = [self itemAtIndexPath:indexPath];
  if (!item || item->control_type != IOSConfigControlType::kToggle) {
    return;
  }
  item->bool_value = sender.isOn;
  [self markPendingChanges];
}

- (void)tableView:(UITableView*)tableView didSelectRowAtIndexPath:(NSIndexPath*)indexPath {
  IOSConfigItem* item = [self itemAtIndexPath:indexPath];
  if (!item || item->control_type == IOSConfigControlType::kToggle) {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    return;
  }

  if (item->control_type == IOSConfigControlType::kAction) {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    switch (item->action) {
      case IOSConfigAction::kViewRecentLog: {
        XeniaLogViewController* log_vc = [[XeniaLogViewController alloc] init];
        [self.navigationController pushViewController:log_vc animated:YES];
      } break;
      case IOSConfigAction::kNone:
      default:
        break;
    }
    return;
  }

  XeniaChoiceListViewController* choice_vc = [[XeniaChoiceListViewController alloc]
      initWithTitle:ToNSString(item->title)
           subtitle:ToNSString(item->subtitle)
            choices:item->choices
      selectedValue:item->choice_value
        onSelection:^(int64_t selected_value) {
          item->choice_value = selected_value;
          [self markPendingChanges];
          [self.tableView reloadRowsAtIndexPaths:@[ indexPath ]
                                withRowAnimation:UITableViewRowAnimationNone];
        }];
  [self.navigationController pushViewController:choice_vc animated:YES];
  [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

- (void)cancelTapped:(id)sender {
  if (!hasPendingChanges_) {
    [self dismissViewControllerAnimated:YES completion:nil];
    return;
  }

  UIAlertController* confirm =
      [UIAlertController alertControllerWithTitle:@"Discard Changes?"
                                          message:@"You have unsaved setting changes."
                                   preferredStyle:UIAlertControllerStyleAlert];
  [confirm addAction:[UIAlertAction actionWithTitle:@"Keep Editing"
                                              style:UIAlertActionStyleCancel
                                            handler:nil]];
  [confirm addAction:[UIAlertAction actionWithTitle:@"Discard"
                                              style:UIAlertActionStyleDestructive
                                            handler:^(__unused UIAlertAction* action) {
                                              [self dismissViewControllerAnimated:YES
                                                                       completion:nil];
                                            }]];
  [self presentViewController:confirm animated:YES completion:nil];
}

- (void)saveTapped:(id)sender {
  BOOL saved = ApplyIOSConfigSections(sections_) ? YES : NO;
  hasPendingChanges_ = NO;
  saveButton_.enabled = NO;

  NSString* title = saved ? @"Settings Saved" : @"Save Completed With Warnings";
  NSString* message = saved ? @"Saved to xenia-edge.config.toml."
                            : @"Some settings could not be applied. Check xenia.log.";
  UIAlertController* alert =
      [UIAlertController alertControllerWithTitle:title
                                          message:message
                                   preferredStyle:UIAlertControllerStyleAlert];
  [alert addAction:[UIAlertAction actionWithTitle:@"Done"
                                            style:UIAlertActionStyleDefault
                                          handler:^(__unused UIAlertAction* action) {
                                            [self dismissViewControllerAnimated:YES completion:nil];
                                          }]];
  [self presentViewController:alert animated:YES completion:nil];
}

@end

// ---------------------------------------------------------------------------
// XeniaViewController - manages the Metal view and game launcher UI.
// ---------------------------------------------------------------------------
@interface XeniaViewController : UIViewController <UIDocumentPickerDelegate,
                                                   UICollectionViewDataSource,
                                                   UICollectionViewDelegateFlowLayout>
@property(nonatomic, strong) XeniaMetalView* metalView;
@property(nonatomic, strong) UIView* launcherOverlay;
@property(nonatomic, strong) UIButton* openGameButton;
@property(nonatomic, strong) UIButton* settingsButton;
@property(nonatomic, strong) UIButton* profileButton;
@property(nonatomic, strong) UILabel* titleLabel;
@property(nonatomic, strong) UILabel* statusLabel;
@property(nonatomic, strong) UILabel* signedInProfileLabel;
@property(nonatomic, strong) UICollectionView* importedGamesCollectionView;
@property(nonatomic, strong) UILabel* importedGamesEmptyLabel;
@property(nonatomic, strong) UIView* inGameMenuOverlay;
@property(nonatomic, assign) BOOL gameRunning;
@property(nonatomic, assign) xe::ui::IOSWindowedAppContext* appContext;

// JIT status widgets.
@property(nonatomic, strong) UIView* jitWarningCard;
@property(nonatomic, strong) UIView* jitStatusDot;
@property(nonatomic, strong) UILabel* jitStatusLabel;
@property(nonatomic, strong) NSTimer* jitPollTimer;
@property(nonatomic, assign) BOOL jitAcquired;
@property(nonatomic, assign) BOOL launcherLandscapeUnlocked;
- (void)refreshSignedInProfileUI;
- (void)presentSystemSigninPromptForUserIndex:(uint32_t)user_index
                                   usersNeeded:(uint32_t)users_needed
                                    completion:(void (^)(BOOL success))completion;
- (void)presentSystemKeyboardPromptWithTitle:(NSString*)title
                                 description:(NSString*)description
                                 defaultText:(NSString*)default_text
                                  completion:(void (^)(BOOL cancelled,
                                                       NSString* text))completion;
- (void)setupInGameMenuOverlay;
- (void)toggleInGameMenuTapped:(UITapGestureRecognizer*)recognizer;
- (void)resumeGameTapped:(UIButton*)sender;
- (void)inGameSettingsTapped:(UIButton*)sender;
- (void)inGameLiveLogTapped:(UIButton*)sender;
- (void)exitGameTapped:(UIButton*)sender;
- (void)hideInGameMenuOverlay;
@end

@implementation XeniaViewController {
  std::vector<IOSDiscoveredGame> discovered_games_;
}

- (void)viewDidLoad {
  [super viewDidLoad];
  self.view.backgroundColor = [UIColor blackColor];
  self.jitAcquired = NO;
  self.launcherLandscapeUnlocked = NO;
  self.gameRunning = NO;

  // Create the Metal-backed rendering view (full screen, behind everything).
  self.metalView = [[XeniaMetalView alloc] initWithFrame:self.view.bounds];
  self.metalView.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.metalView.contentScaleFactor = [UIScreen mainScreen].scale;
  [self.view addSubview:self.metalView];

  // Create the launcher overlay UI immediately. When JIT is missing, keep
  // settings/navigation available but gate game launch with status.
  [self setupLauncherOverlay];
  [self setupInGameMenuOverlay];
  UITapGestureRecognizer* tap = [[UITapGestureRecognizer alloc]
      initWithTarget:self
              action:@selector(toggleInGameMenuTapped:)];
  tap.numberOfTapsRequired = 1;
  tap.cancelsTouchesInView = NO;
  [self.view addGestureRecognizer:tap];
  [self updateJITStatusIndicator];
  [self updateJITAvailabilityUI];
  [self refreshSignedInProfileUI];
  [self refreshImportedGames];

  // Start polling for JIT.
  [self startJITPoll];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
  xe_request_portrait_orientation(self);
  if (!self.launcherLandscapeUnlocked) {
    self.launcherLandscapeUnlocked = YES;
    [self setNeedsUpdateOfSupportedInterfaceOrientations];
  }
}

// ---------------------------------------------------------------------------
// JIT polling -- checks every 0.5s until JIT is available.
// ---------------------------------------------------------------------------
- (void)startJITPoll {
  // Check immediately first.
  if (xe_check_jit_available()) {
    [self onJITAcquired];
    return;
  }

  XELOGI("iOS: JIT not yet available, polling...");
  self.jitPollTimer = [NSTimer scheduledTimerWithTimeInterval:0.5
                                                       target:self
                                                     selector:@selector(pollJIT:)
                                                     userInfo:nil
                                                      repeats:YES];
}

- (void)pollJIT:(NSTimer*)timer {
  if (xe_check_jit_available()) {
    [timer invalidate];
    self.jitPollTimer = nil;
    [self onJITAcquired];
  }
}

- (void)onJITAcquired {
  self.jitAcquired = YES;
  XELOGI("iOS: JIT acquired!");
  [self updateJITStatusIndicator];
  [self updateJITAvailabilityUI];
}

// ---------------------------------------------------------------------------
// Launcher overlay with Open Game button.
// ---------------------------------------------------------------------------
- (void)setupLauncherOverlay {
  self.launcherOverlay = [[UIView alloc] initWithFrame:self.view.bounds];
  self.launcherOverlay.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.launcherOverlay.backgroundColor = [UIColor colorWithWhite:0.0 alpha:0.85];
  [self.view addSubview:self.launcherOverlay];
  UILayoutGuide* safe_guide = self.launcherOverlay.safeAreaLayoutGuide;

  // JIT status indicator (green/red dot + label) at the top.
  UIView* statusContainer = [[UIView alloc] init];
  statusContainer.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:statusContainer];

  self.jitStatusDot = [[UIView alloc] init];
  self.jitStatusDot.translatesAutoresizingMaskIntoConstraints = NO;
  self.jitStatusDot.backgroundColor = [UIColor systemRedColor];
  self.jitStatusDot.layer.cornerRadius = 5;
  [statusContainer addSubview:self.jitStatusDot];

  self.jitStatusLabel = [[UILabel alloc] init];
  self.jitStatusLabel.text = @"JIT Not Detected";
  self.jitStatusLabel.textColor = [UIColor systemRedColor];
  self.jitStatusLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightMedium];
  self.jitStatusLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [statusContainer addSubview:self.jitStatusLabel];

  // Title label.
  self.titleLabel = [[UILabel alloc] init];
  self.titleLabel.text = @"Xenia-Edge";
  self.titleLabel.textColor = [UIColor whiteColor];
  self.titleLabel.font = [UIFont systemFontOfSize:42 weight:UIFontWeightBold];
  self.titleLabel.textAlignment = NSTextAlignmentCenter;
  self.titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:self.titleLabel];

  // Subtitle label.
  UILabel* subtitleLabel = [[UILabel alloc] init];
  subtitleLabel.text = @"Xbox 360 Emulator";
  subtitleLabel.textColor = [UIColor lightGrayColor];
  subtitleLabel.font = [UIFont systemFontOfSize:16 weight:UIFontWeightRegular];
  subtitleLabel.textAlignment = NSTextAlignmentCenter;
  subtitleLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:subtitleLabel];

  // Non-blocking JIT warning card shown until JIT becomes available.
  self.jitWarningCard = [[UIView alloc] init];
  self.jitWarningCard.translatesAutoresizingMaskIntoConstraints = NO;
  self.jitWarningCard.backgroundColor = [UIColor colorWithRed:0.34 green:0.08 blue:0.03 alpha:0.92];
  self.jitWarningCard.layer.cornerRadius = 12;
  [self.launcherOverlay addSubview:self.jitWarningCard];

  UIImageView* jitWarningIcon = [[UIImageView alloc]
      initWithImage:[UIImage systemImageNamed:@"exclamationmark.triangle.fill"]];
  jitWarningIcon.translatesAutoresizingMaskIntoConstraints = NO;
  jitWarningIcon.tintColor = [UIColor systemOrangeColor];
  [self.jitWarningCard addSubview:jitWarningIcon];

  UILabel* jitWarningTitle = [[UILabel alloc] init];
  jitWarningTitle.translatesAutoresizingMaskIntoConstraints = NO;
  jitWarningTitle.text = @"JIT Not Detected";
  jitWarningTitle.textColor = [UIColor colorWithRed:1.0 green:0.88 blue:0.72 alpha:1.0];
  jitWarningTitle.font = [UIFont systemFontOfSize:16 weight:UIFontWeightSemibold];
  [self.jitWarningCard addSubview:jitWarningTitle];

  UILabel* jitWarningBody = [[UILabel alloc] init];
  jitWarningBody.translatesAutoresizingMaskIntoConstraints = NO;
  jitWarningBody.numberOfLines = 0;
  jitWarningBody.text = @"Enable JIT with StikDebug, SideJITServer, or AltJIT. "
                        @"Settings and menu navigation stay available.";
  jitWarningBody.textColor = [UIColor colorWithRed:0.95 green:0.83 blue:0.76 alpha:1.0];
  jitWarningBody.font = [UIFont systemFontOfSize:13 weight:UIFontWeightRegular];
  [self.jitWarningCard addSubview:jitWarningBody];

  // Import button.
  UIButtonConfiguration* config = [UIButtonConfiguration filledButtonConfiguration];
  config.title = @"Import Game";
  config.image = [UIImage systemImageNamed:@"folder"];
  config.imagePadding = 8;
  config.baseBackgroundColor = [UIColor systemGreenColor];
  config.baseForegroundColor = [UIColor whiteColor];
  config.cornerStyle = UIButtonConfigurationCornerStyleLarge;
  config.contentInsets = NSDirectionalEdgeInsetsMake(14, 32, 14, 32);

  self.openGameButton = [UIButton buttonWithConfiguration:config
                                            primaryAction:nil];
  self.openGameButton.translatesAutoresizingMaskIntoConstraints = NO;
  [self.openGameButton addTarget:self
                          action:@selector(openGameTapped:)
                forControlEvents:UIControlEventTouchUpInside];
  [self.launcherOverlay addSubview:self.openGameButton];

  UIButtonConfiguration* settings_config = [UIButtonConfiguration tintedButtonConfiguration];
  settings_config.title = @"Settings";
  settings_config.image = [UIImage systemImageNamed:@"slider.horizontal.3"];
  settings_config.imagePadding = 6;
  settings_config.baseForegroundColor = [UIColor whiteColor];
  settings_config.baseBackgroundColor = [UIColor colorWithWhite:1.0 alpha:0.14];
  settings_config.cornerStyle = UIButtonConfigurationCornerStyleCapsule;
  settings_config.contentInsets = NSDirectionalEdgeInsetsMake(8, 12, 8, 12);

  self.settingsButton = [UIButton buttonWithConfiguration:settings_config primaryAction:nil];
  self.settingsButton.translatesAutoresizingMaskIntoConstraints = NO;
  [self.settingsButton addTarget:self
                          action:@selector(openSettingsTapped:)
                forControlEvents:UIControlEventTouchUpInside];
  [self.launcherOverlay addSubview:self.settingsButton];

  UIButtonConfiguration* profile_config = [UIButtonConfiguration tintedButtonConfiguration];
  profile_config.title = @"Profile";
  profile_config.image = [UIImage systemImageNamed:@"person.circle"];
  profile_config.imagePadding = 6;
  profile_config.baseForegroundColor = [UIColor whiteColor];
  profile_config.baseBackgroundColor = [UIColor colorWithWhite:1.0 alpha:0.14];
  profile_config.cornerStyle = UIButtonConfigurationCornerStyleCapsule;
  profile_config.contentInsets = NSDirectionalEdgeInsetsMake(8, 12, 8, 12);

  self.profileButton = [UIButton buttonWithConfiguration:profile_config primaryAction:nil];
  self.profileButton.translatesAutoresizingMaskIntoConstraints = NO;
  [self.profileButton addTarget:self
                         action:@selector(openProfileTapped:)
               forControlEvents:UIControlEventTouchUpInside];
  [self.launcherOverlay addSubview:self.profileButton];

  // Status label (for showing loading state).
  self.statusLabel = [[UILabel alloc] init];
  self.statusLabel.text = @"";
  self.statusLabel.textColor = [UIColor lightGrayColor];
  self.statusLabel.font =
      [UIFont systemFontOfSize:14 weight:UIFontWeightRegular];
  self.statusLabel.textAlignment = NSTextAlignmentCenter;
  self.statusLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:self.statusLabel];

  self.signedInProfileLabel = [[UILabel alloc] init];
  self.signedInProfileLabel.text = @"";
  self.signedInProfileLabel.textColor = [UIColor colorWithWhite:1.0 alpha:0.88];
  self.signedInProfileLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightSemibold];
  self.signedInProfileLabel.textAlignment = NSTextAlignmentCenter;
  self.signedInProfileLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:self.signedInProfileLabel];

  UILabel* importedGamesTitleLabel = [[UILabel alloc] init];
  importedGamesTitleLabel.translatesAutoresizingMaskIntoConstraints = NO;
  importedGamesTitleLabel.text = @"Imported Games";
  importedGamesTitleLabel.textColor = [UIColor whiteColor];
  importedGamesTitleLabel.font = [UIFont systemFontOfSize:17 weight:UIFontWeightSemibold];
  [self.launcherOverlay addSubview:importedGamesTitleLabel];

  UICollectionViewFlowLayout* layout = [[UICollectionViewFlowLayout alloc] init];
  layout.minimumInteritemSpacing = 12;
  layout.minimumLineSpacing = 12;
  layout.sectionInset = UIEdgeInsetsMake(0, 0, 0, 0);
  self.importedGamesCollectionView = [[UICollectionView alloc] initWithFrame:CGRectZero
                                                        collectionViewLayout:layout];
  self.importedGamesCollectionView.translatesAutoresizingMaskIntoConstraints = NO;
  self.importedGamesCollectionView.dataSource = self;
  self.importedGamesCollectionView.delegate = self;
  self.importedGamesCollectionView.backgroundColor = [UIColor clearColor];
  self.importedGamesCollectionView.alwaysBounceVertical = YES;
  self.importedGamesCollectionView.showsHorizontalScrollIndicator = NO;
  self.importedGamesCollectionView.layer.cornerRadius = 12;
  [self.importedGamesCollectionView registerClass:[XeniaGameTileCell class]
                       forCellWithReuseIdentifier:@"ImportedGameCell"];
  [self.launcherOverlay addSubview:self.importedGamesCollectionView];

  UIView* emptyBackgroundView = [[UIView alloc] initWithFrame:CGRectZero];
  self.importedGamesEmptyLabel = [[UILabel alloc] init];
  self.importedGamesEmptyLabel.translatesAutoresizingMaskIntoConstraints = NO;
  self.importedGamesEmptyLabel.text = @"No imported games found in Documents.";
  self.importedGamesEmptyLabel.textColor = [UIColor colorWithWhite:1.0 alpha:0.65];
  self.importedGamesEmptyLabel.font = [UIFont systemFontOfSize:14 weight:UIFontWeightMedium];
  self.importedGamesEmptyLabel.textAlignment = NSTextAlignmentCenter;
  self.importedGamesEmptyLabel.numberOfLines = 0;
  [emptyBackgroundView addSubview:self.importedGamesEmptyLabel];
  [NSLayoutConstraint activateConstraints:@[
    [self.importedGamesEmptyLabel.centerXAnchor
        constraintEqualToAnchor:emptyBackgroundView.centerXAnchor],
    [self.importedGamesEmptyLabel.centerYAnchor
        constraintEqualToAnchor:emptyBackgroundView.centerYAnchor],
    [self.importedGamesEmptyLabel.leadingAnchor
        constraintGreaterThanOrEqualToAnchor:emptyBackgroundView.leadingAnchor
                                    constant:20],
    [self.importedGamesEmptyLabel.trailingAnchor
        constraintLessThanOrEqualToAnchor:emptyBackgroundView.trailingAnchor
                                 constant:-20],
  ]];
  self.importedGamesCollectionView.backgroundView = emptyBackgroundView;

  // Layout constraints.
  [NSLayoutConstraint activateConstraints:@[
    // JIT status indicator at top center.
    [statusContainer.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [statusContainer.topAnchor constraintEqualToAnchor:safe_guide.topAnchor constant:16],

    [self.jitStatusDot.leadingAnchor constraintEqualToAnchor:statusContainer.leadingAnchor],
    [self.jitStatusDot.centerYAnchor constraintEqualToAnchor:statusContainer.centerYAnchor],
    [self.jitStatusDot.widthAnchor constraintEqualToConstant:10],
    [self.jitStatusDot.heightAnchor constraintEqualToConstant:10],

    [self.jitStatusLabel.leadingAnchor constraintEqualToAnchor:self.jitStatusDot.trailingAnchor
                                                      constant:6],
    [self.jitStatusLabel.trailingAnchor constraintEqualToAnchor:statusContainer.trailingAnchor],
    [self.jitStatusLabel.centerYAnchor constraintEqualToAnchor:statusContainer.centerYAnchor],

    [self.settingsButton.trailingAnchor constraintEqualToAnchor:safe_guide.trailingAnchor
                                                       constant:-16],
    [self.settingsButton.topAnchor constraintEqualToAnchor:safe_guide.topAnchor constant:12],
    [self.profileButton.leadingAnchor constraintEqualToAnchor:safe_guide.leadingAnchor constant:16],
    [self.profileButton.topAnchor constraintEqualToAnchor:safe_guide.topAnchor constant:12],

    // Title, subtitle, and warning card.
    [self.titleLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.titleLabel.topAnchor constraintEqualToAnchor:safe_guide.topAnchor constant:56],
    [subtitleLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [subtitleLabel.topAnchor constraintEqualToAnchor:self.titleLabel.bottomAnchor constant:6],

    [self.jitWarningCard.topAnchor constraintEqualToAnchor:subtitleLabel.bottomAnchor constant:18],
    [self.jitWarningCard.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.jitWarningCard.leadingAnchor constraintGreaterThanOrEqualToAnchor:safe_guide.leadingAnchor
                                                                   constant:20],
    [self.jitWarningCard.trailingAnchor constraintLessThanOrEqualToAnchor:safe_guide.trailingAnchor
                                                                 constant:-20],
    [self.jitWarningCard.widthAnchor constraintLessThanOrEqualToConstant:640],

    [jitWarningIcon.leadingAnchor constraintEqualToAnchor:self.jitWarningCard.leadingAnchor
                                                 constant:14],
    [jitWarningIcon.topAnchor constraintEqualToAnchor:self.jitWarningCard.topAnchor constant:14],
    [jitWarningIcon.widthAnchor constraintEqualToConstant:18],
    [jitWarningIcon.heightAnchor constraintEqualToConstant:18],

    [jitWarningTitle.leadingAnchor constraintEqualToAnchor:jitWarningIcon.trailingAnchor
                                                  constant:10],
    [jitWarningTitle.trailingAnchor constraintEqualToAnchor:self.jitWarningCard.trailingAnchor
                                                   constant:-14],
    [jitWarningTitle.topAnchor constraintEqualToAnchor:self.jitWarningCard.topAnchor constant:12],

    [jitWarningBody.leadingAnchor constraintEqualToAnchor:jitWarningTitle.leadingAnchor],
    [jitWarningBody.trailingAnchor constraintEqualToAnchor:jitWarningTitle.trailingAnchor],
    [jitWarningBody.topAnchor constraintEqualToAnchor:jitWarningTitle.bottomAnchor constant:6],
    [jitWarningBody.bottomAnchor constraintEqualToAnchor:self.jitWarningCard.bottomAnchor
                                                constant:-12],

    [self.openGameButton.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.openGameButton.topAnchor constraintEqualToAnchor:self.jitWarningCard.bottomAnchor
                                                  constant:20],

    [self.statusLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.statusLabel.topAnchor constraintEqualToAnchor:self.openGameButton.bottomAnchor
                                               constant:12],
    [self.statusLabel.leadingAnchor constraintGreaterThanOrEqualToAnchor:safe_guide.leadingAnchor
                                                                constant:20],
    [self.statusLabel.trailingAnchor constraintLessThanOrEqualToAnchor:safe_guide.trailingAnchor
                                                              constant:-20],

    [self.signedInProfileLabel.centerXAnchor
        constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.signedInProfileLabel.topAnchor constraintEqualToAnchor:self.statusLabel.bottomAnchor
                                                        constant:8],
    [self.signedInProfileLabel.leadingAnchor
        constraintGreaterThanOrEqualToAnchor:safe_guide.leadingAnchor
                                    constant:20],
    [self.signedInProfileLabel.trailingAnchor
        constraintLessThanOrEqualToAnchor:safe_guide.trailingAnchor
                                 constant:-20],

    [importedGamesTitleLabel.leadingAnchor constraintEqualToAnchor:safe_guide.leadingAnchor
                                                          constant:18],
    [importedGamesTitleLabel.topAnchor
        constraintEqualToAnchor:self.signedInProfileLabel.bottomAnchor
                       constant:14],

    [self.importedGamesCollectionView.leadingAnchor constraintEqualToAnchor:safe_guide.leadingAnchor
                                                                   constant:14],
    [self.importedGamesCollectionView.trailingAnchor
        constraintEqualToAnchor:safe_guide.trailingAnchor
                       constant:-14],
    [self.importedGamesCollectionView.topAnchor
        constraintEqualToAnchor:importedGamesTitleLabel.bottomAnchor
                       constant:8],
    [self.importedGamesCollectionView.bottomAnchor constraintEqualToAnchor:safe_guide.bottomAnchor
                                                                  constant:-12],
  ]];
}

- (void)setupInGameMenuOverlay {
  self.inGameMenuOverlay = [[UIView alloc] initWithFrame:self.view.bounds];
  self.inGameMenuOverlay.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.inGameMenuOverlay.backgroundColor = [UIColor colorWithWhite:0.0 alpha:0.58];
  self.inGameMenuOverlay.hidden = YES;
  [self.view addSubview:self.inGameMenuOverlay];

  UIView* panel = [[UIView alloc] init];
  panel.translatesAutoresizingMaskIntoConstraints = NO;
  panel.backgroundColor = [UIColor colorWithWhite:0.08 alpha:0.94];
  panel.layer.cornerRadius = 16.0;
  panel.layer.borderWidth = 1.0;
  panel.layer.borderColor = [UIColor colorWithWhite:1.0 alpha:0.08].CGColor;
  [self.inGameMenuOverlay addSubview:panel];

  UILabel* title = [[UILabel alloc] init];
  title.translatesAutoresizingMaskIntoConstraints = NO;
  title.text = @"In-Game Menu";
  title.textColor = [UIColor whiteColor];
  title.font = [UIFont systemFontOfSize:20 weight:UIFontWeightSemibold];
  title.textAlignment = NSTextAlignmentCenter;
  [panel addSubview:title];

  UILabel* subtitle = [[UILabel alloc] init];
  subtitle.translatesAutoresizingMaskIntoConstraints = NO;
  subtitle.text = @"Tap anywhere to close";
  subtitle.textColor = [UIColor colorWithWhite:1.0 alpha:0.68];
  subtitle.font = [UIFont systemFontOfSize:13 weight:UIFontWeightRegular];
  subtitle.textAlignment = NSTextAlignmentCenter;
  [panel addSubview:subtitle];

  UIButtonConfiguration* resume_config = [UIButtonConfiguration filledButtonConfiguration];
  resume_config.title = @"Resume";
  resume_config.baseBackgroundColor = [UIColor systemGreenColor];
  resume_config.baseForegroundColor = [UIColor whiteColor];
  resume_config.cornerStyle = UIButtonConfigurationCornerStyleLarge;
  resume_config.contentInsets = NSDirectionalEdgeInsetsMake(12, 18, 12, 18);
  UIButton* resume = [UIButton buttonWithConfiguration:resume_config primaryAction:nil];
  resume.translatesAutoresizingMaskIntoConstraints = NO;
  [resume addTarget:self action:@selector(resumeGameTapped:) forControlEvents:UIControlEventTouchUpInside];
  [panel addSubview:resume];

  UIButtonConfiguration* settings_config = [UIButtonConfiguration tintedButtonConfiguration];
  settings_config.title = @"Settings";
  settings_config.image = [UIImage systemImageNamed:@"slider.horizontal.3"];
  settings_config.imagePadding = 6;
  settings_config.baseForegroundColor = [UIColor whiteColor];
  settings_config.baseBackgroundColor = [UIColor colorWithWhite:1.0 alpha:0.16];
  settings_config.cornerStyle = UIButtonConfigurationCornerStyleLarge;
  settings_config.contentInsets = NSDirectionalEdgeInsetsMake(10, 16, 10, 16);
  UIButton* settings = [UIButton buttonWithConfiguration:settings_config primaryAction:nil];
  settings.translatesAutoresizingMaskIntoConstraints = NO;
  [settings addTarget:self
               action:@selector(inGameSettingsTapped:)
     forControlEvents:UIControlEventTouchUpInside];
  [panel addSubview:settings];

  UIButtonConfiguration* live_log_config = [UIButtonConfiguration tintedButtonConfiguration];
  live_log_config.title = @"Live Log";
  live_log_config.image = [UIImage systemImageNamed:@"doc.text"];
  live_log_config.imagePadding = 6;
  live_log_config.baseForegroundColor = [UIColor whiteColor];
  live_log_config.baseBackgroundColor = [UIColor colorWithWhite:1.0 alpha:0.16];
  live_log_config.cornerStyle = UIButtonConfigurationCornerStyleLarge;
  live_log_config.contentInsets = NSDirectionalEdgeInsetsMake(10, 16, 10, 16);
  UIButton* live_log = [UIButton buttonWithConfiguration:live_log_config primaryAction:nil];
  live_log.translatesAutoresizingMaskIntoConstraints = NO;
  [live_log addTarget:self
               action:@selector(inGameLiveLogTapped:)
     forControlEvents:UIControlEventTouchUpInside];
  [panel addSubview:live_log];

  UIButtonConfiguration* exit_config = [UIButtonConfiguration tintedButtonConfiguration];
  exit_config.title = @"Exit To Library";
  exit_config.image = [UIImage systemImageNamed:@"rectangle.portrait.and.arrow.right"];
  exit_config.imagePadding = 6;
  exit_config.baseForegroundColor = [UIColor whiteColor];
  exit_config.baseBackgroundColor = [UIColor colorWithRed:0.63 green:0.18 blue:0.18 alpha:0.92];
  exit_config.cornerStyle = UIButtonConfigurationCornerStyleLarge;
  exit_config.contentInsets = NSDirectionalEdgeInsetsMake(10, 16, 10, 16);
  UIButton* exit_button = [UIButton buttonWithConfiguration:exit_config primaryAction:nil];
  exit_button.translatesAutoresizingMaskIntoConstraints = NO;
  [exit_button addTarget:self action:@selector(exitGameTapped:) forControlEvents:UIControlEventTouchUpInside];
  [panel addSubview:exit_button];

  [NSLayoutConstraint activateConstraints:@[
    [panel.centerXAnchor constraintEqualToAnchor:self.inGameMenuOverlay.centerXAnchor],
    [panel.centerYAnchor constraintEqualToAnchor:self.inGameMenuOverlay.centerYAnchor],
    [panel.leadingAnchor constraintGreaterThanOrEqualToAnchor:self.inGameMenuOverlay.safeAreaLayoutGuide.leadingAnchor
                                                     constant:24],
    [panel.trailingAnchor constraintLessThanOrEqualToAnchor:self.inGameMenuOverlay.safeAreaLayoutGuide.trailingAnchor
                                                   constant:-24],
    [panel.widthAnchor constraintLessThanOrEqualToConstant:420],

    [title.topAnchor constraintEqualToAnchor:panel.topAnchor constant:18],
    [title.leadingAnchor constraintEqualToAnchor:panel.leadingAnchor constant:20],
    [title.trailingAnchor constraintEqualToAnchor:panel.trailingAnchor constant:-20],

    [subtitle.topAnchor constraintEqualToAnchor:title.bottomAnchor constant:4],
    [subtitle.leadingAnchor constraintEqualToAnchor:title.leadingAnchor],
    [subtitle.trailingAnchor constraintEqualToAnchor:title.trailingAnchor],

    [resume.topAnchor constraintEqualToAnchor:subtitle.bottomAnchor constant:16],
    [resume.leadingAnchor constraintEqualToAnchor:panel.leadingAnchor constant:14],
    [resume.trailingAnchor constraintEqualToAnchor:panel.trailingAnchor constant:-14],

    [settings.topAnchor constraintEqualToAnchor:resume.bottomAnchor constant:10],
    [settings.leadingAnchor constraintEqualToAnchor:resume.leadingAnchor],
    [settings.trailingAnchor constraintEqualToAnchor:resume.trailingAnchor],

    [live_log.topAnchor constraintEqualToAnchor:settings.bottomAnchor constant:10],
    [live_log.leadingAnchor constraintEqualToAnchor:resume.leadingAnchor],
    [live_log.trailingAnchor constraintEqualToAnchor:resume.trailingAnchor],

    [exit_button.topAnchor constraintEqualToAnchor:live_log.bottomAnchor constant:10],
    [exit_button.leadingAnchor constraintEqualToAnchor:resume.leadingAnchor],
    [exit_button.trailingAnchor constraintEqualToAnchor:resume.trailingAnchor],
    [exit_button.bottomAnchor constraintEqualToAnchor:panel.bottomAnchor constant:-14],
  ]];
}

- (void)toggleInGameMenuTapped:(UITapGestureRecognizer*)recognizer {
  if (recognizer.state != UIGestureRecognizerStateRecognized) {
    return;
  }
  if (self.launcherOverlay.hidden == NO || !self.gameRunning || self.presentedViewController) {
    return;
  }

  BOOL should_show = self.inGameMenuOverlay.hidden;
  if (should_show) {
    self.inGameMenuOverlay.alpha = 0.0;
    self.inGameMenuOverlay.hidden = NO;
    [UIView animateWithDuration:0.18
                     animations:^{
                       self.inGameMenuOverlay.alpha = 1.0;
                     }];
  } else {
    [self hideInGameMenuOverlay];
  }
}

- (void)hideInGameMenuOverlay {
  if (self.inGameMenuOverlay.hidden) {
    return;
  }
  [UIView animateWithDuration:0.15
      animations:^{
        self.inGameMenuOverlay.alpha = 0.0;
      }
      completion:^(__unused BOOL finished) {
        self.inGameMenuOverlay.hidden = YES;
        self.inGameMenuOverlay.alpha = 1.0;
      }];
}

- (void)resumeGameTapped:(UIButton*)sender {
  [self hideInGameMenuOverlay];
}

- (void)inGameSettingsTapped:(UIButton*)sender {
  [self hideInGameMenuOverlay];
  [self openSettingsTapped:nil];
}

- (void)inGameLiveLogTapped:(UIButton*)sender {
  [self hideInGameMenuOverlay];
  XeniaLogViewController* log_vc = [[XeniaLogViewController alloc] init];
  XeniaLandscapeNavigationController* nav =
      [[XeniaLandscapeNavigationController alloc] initWithRootViewController:log_vc];
  if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad) {
    nav.modalPresentationStyle = UIModalPresentationFormSheet;
    if (@available(iOS 15.0, *)) {
      UISheetPresentationController* sheet = nav.sheetPresentationController;
      sheet.detents = @[ [UISheetPresentationControllerDetent mediumDetent],
                         [UISheetPresentationControllerDetent largeDetent] ];
      sheet.prefersGrabberVisible = YES;
    }
  } else {
    nav.modalPresentationStyle = UIModalPresentationFullScreen;
  }
  [self presentViewController:nav animated:YES completion:nil];
}

- (void)exitGameTapped:(UIButton*)sender {
  [self hideInGameMenuOverlay];
  BOOL requested_stop = self.appContext ? self.appContext->TerminateCurrentGame() : NO;
  if (requested_stop) {
    [self showLauncherOverlay];
    self.statusLabel.text = @"Stopping game...";
  } else {
    self.statusLabel.text = @"No active game to stop.";
  }
}

- (void)refreshSignedInProfileUI {
  if (!self.appContext) {
    self.profileButton.enabled = NO;
    self.profileButton.alpha = 0.5;
    self.signedInProfileLabel.text = @"Profile system unavailable";
    return;
  }

  self.profileButton.enabled = YES;
  self.profileButton.alpha = 1.0;

  const auto profiles = self.appContext->ListProfiles();
  const xe::ui::IOSProfileSummary* signed_in_profile = nullptr;
  for (const auto& profile : profiles) {
    if (profile.signed_in) {
      signed_in_profile = &profile;
      break;
    }
  }

  if (signed_in_profile) {
    self.signedInProfileLabel.text =
        [NSString stringWithFormat:@"Signed in: %@", ToNSString(signed_in_profile->gamertag)];
  } else if (profiles.empty()) {
    self.signedInProfileLabel.text = @"No local profile yet";
  } else {
    self.signedInProfileLabel.text = @"No profile signed in";
  }
}

- (void)presentProfileCreateAlert {
  UIAlertController* create_alert =
      [UIAlertController alertControllerWithTitle:@"Create Profile"
                                          message:@"Enter a gamertag (1-15 characters)."
                                   preferredStyle:UIAlertControllerStyleAlert];
  [create_alert addTextFieldWithConfigurationHandler:^(UITextField* text_field) {
    text_field.placeholder = @"Gamertag";
    text_field.autocapitalizationType = UITextAutocapitalizationTypeWords;
    text_field.autocorrectionType = UITextAutocorrectionTypeNo;
    text_field.clearButtonMode = UITextFieldViewModeWhileEditing;
  }];
  [create_alert addAction:[UIAlertAction actionWithTitle:@"Cancel"
                                                   style:UIAlertActionStyleCancel
                                                 handler:nil]];
  [create_alert
      addAction:
          [UIAlertAction
              actionWithTitle:@"Create"
                        style:UIAlertActionStyleDefault
                      handler:^(__unused UIAlertAction* action) {
                        UITextField* text_field = create_alert.textFields.firstObject;
                        NSString* raw_text = text_field.text ?: @"";
                        NSString* trimmed = [raw_text
                            stringByTrimmingCharactersInSet:[NSCharacterSet
                                                                whitespaceAndNewlineCharacterSet]];
                        if (trimmed.length == 0 || !self.appContext) {
                          return;
                        }
                        uint64_t xuid =
                            self.appContext->CreateProfile(std::string([trimmed UTF8String]));
                        if (!xuid) {
                          UIAlertController* failure = [UIAlertController
                              alertControllerWithTitle:@"Profile Not Created"
                                               message:
                                                   @"Gamertag is invalid or could not be created."
                                        preferredStyle:UIAlertControllerStyleAlert];
                          [failure addAction:[UIAlertAction actionWithTitle:@"OK"
                                                                      style:UIAlertActionStyleCancel
                                                                    handler:nil]];
                          [self presentViewController:failure animated:YES completion:nil];
                          return;
                        }
                        self.appContext->SignInProfile(xuid);
                        [self refreshSignedInProfileUI];
                        self.statusLabel.text =
                            [NSString stringWithFormat:@"Signed in as %@.", trimmed];
                      }]];
  [self presentViewController:create_alert animated:YES completion:nil];
}

- (void)openProfileTapped:(UIButton*)sender {
  if (!self.appContext) {
    return;
  }

  auto profiles = self.appContext->ListProfiles();
  UIAlertController* sheet =
      [UIAlertController alertControllerWithTitle:@"Profiles"
                                          message:@"Create a profile or sign in."
                                   preferredStyle:UIAlertControllerStyleActionSheet];

  [sheet addAction:[UIAlertAction actionWithTitle:@"Create Profile"
                                            style:UIAlertActionStyleDefault
                                          handler:^(__unused UIAlertAction* action) {
                                            [self presentProfileCreateAlert];
                                          }]];

  for (const auto& profile : profiles) {
    NSString* gamertag = ToNSString(profile.gamertag);
    NSString* title = gamertag;
    if (profile.signed_in) {
      title = [title stringByAppendingString:@" (Signed In)"];
    }
    uint64_t xuid = profile.xuid;
    [sheet addAction:[UIAlertAction actionWithTitle:title
                                              style:UIAlertActionStyleDefault
                                            handler:^(__unused UIAlertAction* action) {
                                              if (!self.appContext) {
                                                return;
                                              }
                                              if (self.appContext->SignInProfile(xuid)) {
                                                [self refreshSignedInProfileUI];
                                                self.statusLabel.text = [NSString
                                                    stringWithFormat:@"Signed in as %@.", gamertag];
                                              }
                                            }]];
  }

  [sheet addAction:[UIAlertAction actionWithTitle:@"Cancel"
                                            style:UIAlertActionStyleCancel
                                          handler:nil]];

  UIPopoverPresentationController* popover = sheet.popoverPresentationController;
  if (popover) {
    popover.sourceView = sender ?: self.profileButton;
    popover.sourceRect = (sender ?: self.profileButton).bounds;
  }

  [self presentViewController:sheet animated:YES completion:nil];
}

- (void)presentSystemSigninPromptForUserIndex:(uint32_t)user_index
                                   usersNeeded:(uint32_t)users_needed
                                    completion:(void (^)(BOOL success))completion {
  if (!self.appContext) {
    if (completion) {
      completion(NO);
    }
    return;
  }

  __block BOOL finished = NO;
  void (^finish)(BOOL) = ^(BOOL success) {
    if (finished) {
      return;
    }
    finished = YES;
    [self refreshSignedInProfileUI];
    if (completion) {
      completion(success);
    }
  };

  auto profiles = self.appContext->ListProfiles();
  __unsafe_unretained XeniaViewController* weak_self = self;
  void (^present_create_alert)(void) = ^{
    XeniaViewController* strong_self = weak_self;
    if (!strong_self || !strong_self.appContext) {
      finish(NO);
      return;
    }

    UIAlertController* create_alert =
        [UIAlertController alertControllerWithTitle:@"Create Profile"
                                            message:@"Enter a gamertag (1-15 characters)."
                                     preferredStyle:UIAlertControllerStyleAlert];
    [create_alert addTextFieldWithConfigurationHandler:^(UITextField* text_field) {
      text_field.placeholder = @"Gamertag";
      text_field.autocapitalizationType = UITextAutocapitalizationTypeWords;
      text_field.autocorrectionType = UITextAutocorrectionTypeNo;
      text_field.clearButtonMode = UITextFieldViewModeWhileEditing;
    }];
    [create_alert addAction:[UIAlertAction actionWithTitle:@"Cancel"
                                                     style:UIAlertActionStyleCancel
                                                   handler:^(__unused UIAlertAction* action) {
                                                     finish(NO);
                                                   }]];
    [create_alert
        addAction:[UIAlertAction
                      actionWithTitle:@"Create"
                                style:UIAlertActionStyleDefault
                              handler:^(__unused UIAlertAction* action) {
                                UITextField* text_field = create_alert.textFields.firstObject;
                                NSString* raw_text = text_field.text ?: @"";
                                NSString* trimmed = [raw_text stringByTrimmingCharactersInSet:
                                                                 [NSCharacterSet whitespaceAndNewlineCharacterSet]];
                                if (trimmed.length == 0 || !strong_self.appContext) {
                                  finish(NO);
                                  return;
                                }
                                uint64_t xuid = strong_self.appContext->CreateProfile(
                                    std::string([trimmed UTF8String]));
                                if (!xuid) {
                                  finish(NO);
                                  return;
                                }
                                BOOL signed_in = strong_self.appContext->SignInProfile(xuid);
                                if (signed_in) {
                                  strong_self.statusLabel.text =
                                      [NSString stringWithFormat:@"Signed in as %@.", trimmed];
                                }
                                finish(signed_in);
                              }]];

    UIViewController* presenter = strong_self;
    while (presenter.presentedViewController) {
      presenter = presenter.presentedViewController;
    }
    [presenter presentViewController:create_alert animated:YES completion:nil];
  };

  if (profiles.empty()) {
    present_create_alert();
    return;
  }

  NSString* message = [NSString stringWithFormat:@"Select profile (needs %u user%@).",
                                                 users_needed, users_needed == 1 ? @"" : @"s"];
  UIAlertController* sheet =
      [UIAlertController alertControllerWithTitle:@"Select Profile"
                                          message:message
                                   preferredStyle:UIAlertControllerStyleActionSheet];

  [sheet addAction:[UIAlertAction actionWithTitle:@"Create Profile"
                                            style:UIAlertActionStyleDefault
                                          handler:^(__unused UIAlertAction* action) {
                                            present_create_alert();
                                          }]];

  for (const auto& profile : profiles) {
    NSString* gamertag = ToNSString(profile.gamertag);
    NSString* title = gamertag;
    if (profile.signed_in) {
      title = [title stringByAppendingString:@" (Signed In)"];
    }
    uint64_t xuid = profile.xuid;
    [sheet addAction:[UIAlertAction actionWithTitle:title
                                              style:UIAlertActionStyleDefault
                                            handler:^(__unused UIAlertAction* action) {
                                              if (!self.appContext) {
                                                finish(NO);
                                                return;
                                              }
                                              BOOL signed_in = self.appContext->SignInProfile(xuid);
                                              finish(signed_in);
                                            }]];
  }

  [sheet addAction:[UIAlertAction actionWithTitle:@"Cancel"
                                            style:UIAlertActionStyleCancel
                                          handler:^(__unused UIAlertAction* action) {
                                            finish(NO);
                                          }]];

  UIPopoverPresentationController* popover = sheet.popoverPresentationController;
  if (popover) {
    popover.sourceView = self.view;
    popover.sourceRect = CGRectMake(CGRectGetMidX(self.view.bounds), CGRectGetMidY(self.view.bounds),
                                    1.0, 1.0);
    popover.permittedArrowDirections = 0;
  }

  UIViewController* presenter = self;
  while (presenter.presentedViewController) {
    presenter = presenter.presentedViewController;
  }
  [presenter presentViewController:sheet animated:YES completion:nil];
}

- (void)presentSystemKeyboardPromptWithTitle:(NSString*)title
                                 description:(NSString*)description
                                 defaultText:(NSString*)default_text
                                  completion:(void (^)(BOOL cancelled,
                                                       NSString* text))completion {
  UIAlertController* alert = [UIAlertController alertControllerWithTitle:title.length
                                                                       ? title
                                                                       : @"Input Required"
                                                                  message:description
                                                           preferredStyle:UIAlertControllerStyleAlert];
  [alert addTextFieldWithConfigurationHandler:^(UITextField* text_field) {
    text_field.text = default_text ?: @"";
    text_field.autocapitalizationType = UITextAutocapitalizationTypeNone;
    text_field.autocorrectionType = UITextAutocorrectionTypeNo;
    text_field.clearButtonMode = UITextFieldViewModeWhileEditing;
  }];

  [alert addAction:[UIAlertAction actionWithTitle:@"Cancel"
                                            style:UIAlertActionStyleCancel
                                          handler:^(__unused UIAlertAction* action) {
                                            if (completion) {
                                              completion(YES, @"");
                                            }
                                          }]];
  [alert addAction:[UIAlertAction actionWithTitle:@"OK"
                                            style:UIAlertActionStyleDefault
                                          handler:^(__unused UIAlertAction* action) {
                                            UITextField* text_field = alert.textFields.firstObject;
                                            NSString* text = text_field.text ?: @"";
                                            if (completion) {
                                              completion(NO, text);
                                            }
                                          }]];

  UIViewController* presenter = self;
  while (presenter.presentedViewController) {
    presenter = presenter.presentedViewController;
  }
  [presenter presentViewController:alert animated:YES completion:nil];
}

- (void)updateJITStatusIndicator {
  if (self.jitAcquired) {
    self.jitStatusDot.backgroundColor = [UIColor systemGreenColor];
    self.jitStatusLabel.text = @"JIT Enabled";
    self.jitStatusLabel.textColor = [UIColor systemGreenColor];
  } else {
    self.jitStatusDot.backgroundColor = [UIColor systemRedColor];
    self.jitStatusLabel.text = @"JIT Not Detected";
    self.jitStatusLabel.textColor = [UIColor systemRedColor];
  }
}

- (void)updateJITAvailabilityUI {
  BOOL jit_ready = self.jitAcquired;
  self.jitWarningCard.hidden = jit_ready;
  self.openGameButton.enabled = YES;
  self.openGameButton.alpha = 1.0;
  if (!jit_ready && self.statusLabel.text.length == 0) {
    self.statusLabel.text = @"JIT not detected yet. Import games or open Settings while waiting.";
  } else if (jit_ready && [self.statusLabel.text hasPrefix:@"JIT not detected yet."]) {
    self.statusLabel.text = @"";
  }
}

- (std::filesystem::path)importedGamesDirectory {
  return xe_get_ios_documents_path() / "games";
}

- (std::filesystem::path)importGameIntoLibrary:(NSURL*)source_url error:(NSError**)error {
  std::filesystem::path source_path([source_url.path UTF8String]);
  std::filesystem::path library_path = [self importedGamesDirectory];

  std::error_code ec;
  std::filesystem::create_directories(library_path, ec);
  if (ec) {
    if (error) {
      *error = [NSError
          errorWithDomain:@"XeniaIOSImport"
                     code:1001
                 userInfo:@{
                   NSLocalizedDescriptionKey : [NSString
                       stringWithFormat:@"Failed creating library folder: %s", ec.message().c_str()]
                 }];
    }
    return {};
  }

  auto weak_source = std::filesystem::weakly_canonical(source_path, ec);
  auto weak_library = std::filesystem::weakly_canonical(library_path, ec);
  if (!ec && weak_source.native().rfind(weak_library.native(), 0) == 0) {
    return weak_source;
  }

  std::filesystem::path destination = library_path / source_path.filename();
  std::filesystem::path stem = destination.stem();
  std::filesystem::path extension = destination.extension();
  for (int attempt = 2; std::filesystem::exists(destination); ++attempt) {
    destination =
        library_path / std::filesystem::path(stem.string() + " (" + std::to_string(attempt) + ")" +
                                             extension.string());
  }

  NSString* source_ns = source_url.path;
  NSString* destination_ns = ToNSString(destination.string());
  if (![[NSFileManager defaultManager] copyItemAtPath:source_ns
                                               toPath:destination_ns
                                                error:error]) {
    return {};
  }
  return destination;
}

- (void)refreshImportedGames {
  discovered_games_.clear();

  std::vector<std::filesystem::path> scan_roots;
  const std::filesystem::path documents_root = xe_get_ios_documents_path();
  const std::filesystem::path library_root = [self importedGamesDirectory];
  scan_roots.push_back(library_root);
  if (documents_root != library_root) {
    scan_roots.push_back(documents_root);
  }

  std::set<std::filesystem::path> seen_paths;
  std::set<uint32_t> seen_god_title_ids;
  for (const auto& root : scan_roots) {
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
      continue;
    }

    std::filesystem::recursive_directory_iterator it(
        root, std::filesystem::directory_options::skip_permission_denied, ec);
    std::filesystem::recursive_directory_iterator end;
    while (!ec && it != end) {
      const auto& entry = *it;
      const auto filename = entry.path().filename().string();
      const auto filename_lower = ToLowerAsciiCopy(filename);
      if (entry.is_directory(ec)) {
        if (filename_lower == "cache" || filename_lower == "cache_host") {
          it.disable_recursion_pending();
        }
      } else if (entry.is_regular_file(ec) &&
                 (IsISOPath(entry.path()) || IsDefaultXexPath(entry.path()) ||
                  IsLikelyGodPath(entry.path()))) {
        const std::filesystem::path canonical_path =
            std::filesystem::weakly_canonical(entry.path(), ec);
        const std::filesystem::path unique_path =
            ec ? std::filesystem::absolute(entry.path(), ec) : canonical_path;
        ec.clear();

        if (seen_paths.insert(unique_path).second) {
          IOSDiscoveredGame game;
          if (!BuildDiscoveredGameFromPath(unique_path, &game)) {
            ++it;
            continue;
          }
          if (IsLikelyGodPath(unique_path) && game.title_id &&
              !seen_god_title_ids.insert(game.title_id).second) {
            ++it;
            continue;
          }
          discovered_games_.push_back(std::move(game));
        }
      }

      ++it;
    }
  }

  std::sort(discovered_games_.begin(), discovered_games_.end(),
            [](const IOSDiscoveredGame& a, const IOSDiscoveredGame& b) {
              if (a.title == b.title) {
                return a.path.filename().string() < b.path.filename().string();
              }
              return a.title < b.title;
            });

  [self.importedGamesCollectionView reloadData];
  self.importedGamesEmptyLabel.hidden = !discovered_games_.empty();
}

- (void)presentJITRequiredAlert {
  UIAlertController* alert = [UIAlertController
      alertControllerWithTitle:@"JIT Not Detected"
                       message:@"Enable JIT first (StikDebug, SideJITServer, or AltJIT). "
                               @"You can open Settings while waiting."
                preferredStyle:UIAlertControllerStyleAlert];
  [alert addAction:[UIAlertAction actionWithTitle:@"Open Settings"
                                            style:UIAlertActionStyleDefault
                                          handler:^(__unused UIAlertAction* action) {
                                            [self openSettingsTapped:nil];
                                          }]];
  [alert addAction:[UIAlertAction actionWithTitle:@"OK"
                                            style:UIAlertActionStyleCancel
                                          handler:nil]];
  [self presentViewController:alert animated:YES completion:nil];
}

- (void)launchGameAtPath:(const std::filesystem::path&)game_path
             displayName:(NSString*)display_name {
  if (!self.jitAcquired) {
    [self presentJITRequiredAlert];
    return;
  }

  NSString* path_ns = ToNSString(game_path.string());
  NSString* fallback_name = ToNSString(game_path.filename().string());
  NSString* game_label = display_name.length ? display_name : fallback_name;
  self.statusLabel.text = [NSString stringWithFormat:@"Loading: %@", game_label];
  self.gameRunning = YES;

  xe_request_landscape_orientation(self);
  [UIView animateWithDuration:0.3
      animations:^{
        self.launcherOverlay.alpha = 0.0;
      }
      completion:^(__unused BOOL finished) {
        self.launcherOverlay.hidden = YES;
      }];

  if (self.appContext) {
    self.appContext->LaunchGame(std::string([path_ns UTF8String]));
  } else {
    self.statusLabel.text = @"Unable to launch game (app context unavailable).";
    self.launcherOverlay.hidden = NO;
    self.launcherOverlay.alpha = 1.0;
  }
}

#pragma mark - UICollectionViewDataSource

- (NSInteger)collectionView:(UICollectionView* __unused)collectionView
     numberOfItemsInSection:(NSInteger)__unused section {
  return static_cast<NSInteger>(discovered_games_.size());
}

- (UICollectionViewCell*)collectionView:(UICollectionView*)collectionView
                 cellForItemAtIndexPath:(NSIndexPath*)indexPath {
  XeniaGameTileCell* cell =
      [collectionView dequeueReusableCellWithReuseIdentifier:@"ImportedGameCell"
                                                forIndexPath:indexPath];
  if (indexPath.item < 0 || static_cast<size_t>(indexPath.item) >= discovered_games_.size()) {
    cell.titleLabel.text = @"";
    cell.iconView.image = nil;
    return cell;
  }

  const IOSDiscoveredGame& game = discovered_games_[static_cast<size_t>(indexPath.item)];
  NSString* title =
      game.title.empty() ? ToNSString(game.path.stem().string()) : ToNSString(game.title);
  cell.titleLabel.text = title;

  UIImage* icon = nil;
  if (!game.icon_data.empty()) {
    NSData* data = [NSData dataWithBytes:game.icon_data.data() length:game.icon_data.size()];
    icon = [UIImage imageWithData:data];
  }
  if (!icon) {
    icon = [UIImage imageNamed:@"128"];
  }
  if (!icon) {
    icon = [UIImage systemImageNamed:@"gamecontroller.fill"];
  }
  cell.iconView.image = icon;
  return cell;
}

#pragma mark - UICollectionViewDelegate

- (void)collectionView:(UICollectionView*)collectionView
    didSelectItemAtIndexPath:(NSIndexPath*)indexPath {
  [collectionView deselectItemAtIndexPath:indexPath animated:YES];
  if (indexPath.item < 0 || static_cast<size_t>(indexPath.item) >= discovered_games_.size()) {
    return;
  }
  const IOSDiscoveredGame& game = discovered_games_[static_cast<size_t>(indexPath.item)];
  [self launchGameAtPath:game.path displayName:ToNSString(game.title)];
}

#pragma mark - UICollectionViewDelegateFlowLayout

- (CGSize)collectionView:(UICollectionView*)collectionView
                    layout:(UICollectionViewLayout* __unused)collectionViewLayout
    sizeForItemAtIndexPath:(NSIndexPath* __unused)indexPath {
  CGFloat content_width = collectionView.bounds.size.width;
  NSInteger columns = 3;
  if (content_width >= 1100) {
    columns = 6;
  } else if (content_width >= 900) {
    columns = 5;
  } else if (content_width >= 680) {
    columns = 4;
  }
  CGFloat spacing = 12.0;
  CGFloat total_spacing = spacing * (columns - 1);
  CGFloat tile_width = floor((content_width - total_spacing) / columns);
  tile_width = MAX(tile_width, 96.0);
  return CGSizeMake(tile_width, tile_width + 40.0);
}

- (void)viewDidLayoutSubviews {
  [super viewDidLayoutSubviews];
  // Notify the app context that the layout changed, so the window and
  // presenter can update for rotation, split-view, or safe-area changes.
  if (self.appContext) {
    self.appContext->NotifyLayoutChanged();
  }
}

- (void)openGameTapped:(UIButton*)sender {
  NSArray<UTType*>* contentTypes = @[
    [UTType typeWithFilenameExtension:@"iso"],
    [UTType typeWithFilenameExtension:@"xex"],
    UTTypeData,
  ];

  UIDocumentPickerViewController* picker =
      [[UIDocumentPickerViewController alloc]
          initForOpeningContentTypes:contentTypes];
  picker.delegate = self;
  picker.allowsMultipleSelection = NO;
  picker.shouldShowFileExtensions = YES;
  [self presentViewController:picker animated:YES completion:nil];
}

- (void)openSettingsTapped:(UIButton*)sender {
  XeniaConfigViewController* settings_vc =
      [[XeniaConfigViewController alloc] initWithStyle:UITableViewStyleInsetGrouped];
  XeniaLandscapeNavigationController* nav =
      [[XeniaLandscapeNavigationController alloc] initWithRootViewController:settings_vc];
  if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad) {
    nav.modalPresentationStyle = UIModalPresentationFormSheet;
    if (@available(iOS 15.0, *)) {
      UISheetPresentationController* sheet = nav.sheetPresentationController;
      sheet.detents = @[
        [UISheetPresentationControllerDetent mediumDetent],
        [UISheetPresentationControllerDetent largeDetent]
      ];
      sheet.prefersGrabberVisible = YES;
    }
  } else {
    nav.modalPresentationStyle = UIModalPresentationFullScreen;
  }
  [self presentViewController:nav animated:YES completion:nil];
}

#pragma mark - UIDocumentPickerDelegate

- (void)documentPicker:(UIDocumentPickerViewController* __unused)controller
    didPickDocumentsAtURLs:(NSArray<NSURL*>*)urls {
  if (urls.count == 0) return;

  NSURL* url = urls[0];
  BOOL access_granted = [url startAccessingSecurityScopedResource];
  XELOGI("iOS: User selected game file: {} (security-scoped: {})", [url.path UTF8String],
         access_granted ? "yes" : "no");

  NSError* import_error = nil;
  std::filesystem::path imported_path = [self importGameIntoLibrary:url error:&import_error];
  if (access_granted) {
    [url stopAccessingSecurityScopedResource];
  }

  if (imported_path.empty()) {
    NSString* message = import_error.localizedDescription ?: @"Failed to import selected game.";
    UIAlertController* alert =
        [UIAlertController alertControllerWithTitle:@"Import Failed"
                                            message:message
                                     preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"OK"
                                              style:UIAlertActionStyleCancel
                                            handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
    return;
  }

  [self refreshImportedGames];
  NSString* imported_name = ToNSString(imported_path.filename().string());
  if (self.jitAcquired) {
    [self launchGameAtPath:imported_path displayName:imported_name];
  } else {
    self.statusLabel.text =
        [NSString stringWithFormat:@"Imported %@. Waiting for JIT.", imported_name];
  }
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController* __unused)controller {
  XELOGI("iOS: Document picker cancelled");
}

#pragma mark - Status bar / home indicator

- (BOOL)prefersStatusBarHidden {
  return YES;
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
  if (self.launcherOverlay.hidden) {
    return UIInterfaceOrientationMaskLandscape;
  }
  if (!self.launcherLandscapeUnlocked) {
    return UIInterfaceOrientationMaskPortrait;
  }
  return UIInterfaceOrientationMaskAllButUpsideDown;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationPortrait;
}

- (BOOL)shouldAutorotate {
  return YES;
}

- (BOOL)prefersHomeIndicatorAutoHidden {
  return YES;
}

- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures {
  return UIRectEdgeAll;
}

#pragma mark - Public API

- (void)showLauncherOverlay {
  self.gameRunning = NO;
  [self hideInGameMenuOverlay];
  self.launcherOverlay.hidden = NO;
  self.statusLabel.text = @"";
  [self refreshImportedGames];
  [self refreshSignedInProfileUI];
  [self updateJITStatusIndicator];
  [self updateJITAvailabilityUI];
  xe_request_portrait_orientation(self);
  [UIView animateWithDuration:0.3
                   animations:^{
                     self.launcherOverlay.alpha = 1.0;
                   }];
}

- (void)dealloc {
  [self.jitPollTimer invalidate];
}

@end

// ---------------------------------------------------------------------------
// XeniaAppDelegate - UIKit application lifecycle.
// ---------------------------------------------------------------------------
@interface XeniaAppDelegate : UIResponder <UIApplicationDelegate>
@property(nonatomic, strong) UIWindow* window;
@end

@implementation XeniaAppDelegate {
  std::unique_ptr<xe::ui::IOSWindowedAppContext> app_context_;
  std::unique_ptr<xe::ui::WindowedApp> app_;
}

- (BOOL)application:(UIApplication*)application
    didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
  // Initialize cvars with no arguments on iOS (arguments come from config).
  int argc = 1;
  char arg0[] = "xenia_edge";
  char* argv[] = {arg0};
  char** argv_ptr = argv;
  cvar::ParseLaunchArguments(argc, argv_ptr, "", {});

  // Create the app context.
  app_context_ = std::make_unique<xe::ui::IOSWindowedAppContext>();

  // Set up the UIKit window and view controller FIRST, so the Metal view
  // is available when the app initializes.
  self.window = [[UIWindow alloc]
      initWithFrame:[[UIScreen mainScreen] bounds]];
  XeniaViewController* vc = [[XeniaViewController alloc] init];
  self.window.rootViewController = vc;
  [self.window makeKeyAndVisible];
  xe_request_portrait_orientation(vc);

  // Force layout so the Metal view is created.
  [vc.view layoutIfNeeded];

  // Store the Metal view and view controller in the app context for
  // iOSWindow to use.
  app_context_->set_metal_view(vc.metalView);
  app_context_->set_view_controller(vc);
  vc.appContext = app_context_.get();
  app_context_->set_signin_ui_prompt_callback(
      [vc](uint32_t user_index, uint32_t users_needed) {
        if ([NSThread isMainThread]) {
          return false;
        }
        __block BOOL success = NO;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        dispatch_async(dispatch_get_main_queue(), ^{
          [vc presentSystemSigninPromptForUserIndex:user_index
                                        usersNeeded:users_needed
                                         completion:^(BOOL prompt_success) {
                                           success = prompt_success;
                                           dispatch_semaphore_signal(sem);
                                         }];
        });
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        return success ? true : false;
      });
  app_context_->set_keyboard_prompt_callback(
      [vc](const std::string& title, const std::string& description,
           const std::string& default_text, std::string* text_out,
           bool* cancelled_out) {
        if ([NSThread isMainThread]) {
          if (cancelled_out) {
            *cancelled_out = true;
          }
          return false;
        }
        __block BOOL cancelled = YES;
        __block NSString* typed_text = @"";
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        dispatch_async(dispatch_get_main_queue(), ^{
          [vc presentSystemKeyboardPromptWithTitle:ToNSString(title)
                                       description:ToNSString(description)
                                       defaultText:ToNSString(default_text)
                                        completion:^(BOOL prompt_cancelled,
                                                     NSString* prompt_text) {
                                          cancelled = prompt_cancelled;
                                          typed_text = prompt_text ?: @"";
                                          dispatch_semaphore_signal(sem);
                                        }];
        });
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        if (text_out) {
          *text_out = std::string([typed_text UTF8String]);
        }
        if (cancelled_out) {
          *cancelled_out = cancelled ? true : false;
        }
        return true;
      });
  app_context_->set_game_exited_callback([vc]() { [vc showLauncherOverlay]; });

  XELOGI("iOS: Metal view ready ({}x{})",
         static_cast<uint32_t>(vc.metalView.bounds.size.width *
                               vc.metalView.contentScaleFactor),
         static_cast<uint32_t>(vc.metalView.bounds.size.height *
                               vc.metalView.contentScaleFactor));

  // Create and initialize the Xenia app.
  app_ = xe::ui::GetWindowedAppCreator()(*app_context_);
  if (cvars::log_file.empty()) {
    cvars::log_file = xe_get_ios_documents_path() / "xenia.log";
  }
  xe::InitializeLogging(app_->GetName(), true);

  if (!app_->OnInitialize()) {
    XELOGE("iOS: App initialization failed");
    return NO;
  }

  [vc refreshSignedInProfileUI];

  XELOGI("iOS: Application launched successfully");
  return YES;
}

- (UIInterfaceOrientationMask)application:(UIApplication*)application
    supportedInterfaceOrientationsForWindow:(UIWindow*)window {
  UIViewController* root = window.rootViewController;
  if (root) {
    return [root supportedInterfaceOrientations];
  }
  return UIInterfaceOrientationMaskPortrait;
}

- (void)applicationWillTerminate:(UIApplication*)application {
  XELOGI("iOS lifecycle: applicationWillTerminate");
  if (app_) {
    app_->InvokeOnDestroy();
    app_.reset();
  }
  app_context_.reset();
}

@end

// ---------------------------------------------------------------------------
// iOS entry point.
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (xe_enable_ptrace_jit_hack()) {
    NSLog(@"iOS: Ptrace JIT setup complete.");
  } else {
    NSLog(@"iOS: Ptrace JIT setup unavailable; use StikDebug/AltJIT/SideJIT.");
  }
  @autoreleasepool {
    return UIApplicationMain(argc, argv, nil,
                             NSStringFromClass([XeniaAppDelegate class]));
  }
}
