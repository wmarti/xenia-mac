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
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/config.h"
#include "xenia/ui/surface_ios.h"
#include "xenia/ui/windowed_app.h"
#include "xenia/ui/windowed_app_context_ios.h"

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

static void xe_request_landscape_orientation(UIViewController* view_controller) {
  if (!view_controller) {
    return;
  }
#if !TARGET_OS_TV
  [view_controller setNeedsUpdateOfSupportedInterfaceOrientations];
  [[UIDevice currentDevice] setValue:@(UIInterfaceOrientationLandscapeRight) forKey:@"orientation"];
  [UIViewController attemptRotationToDeviceOrientation];
#endif
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
                 "Reduces stutter while pipelines compile in background.", true);
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
  AddActionSetting(diagnostics.items, IOSConfigAction::kViewRecentLog, "View Recent Log",
                   "Open the most recent xenia.log output.");
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

- (NSInteger)tableView:(UITableView*)tableView numberOfRowsInSection:(NSInteger)section {
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
  return UIInterfaceOrientationMaskLandscape;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationLandscapeLeft;
}

@end

@implementation XeniaLogViewController {
  UITextView* textView_;
  UILabel* footerLabel_;
}

- (void)viewDidLoad {
  [super viewDidLoad];
  self.title = @"Recent Log";
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
  [self reloadLog];
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
  return UIInterfaceOrientationMaskLandscape;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationLandscapeLeft;
}

- (void)reloadLogTapped:(id)sender {
  [self reloadLog];
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
  std::filesystem::path log_path = GetLogFilePath();
  std::string content = ReadFileTail(log_path, kMaxLogBytes);
  NSString* log_path_ns = ToNSString(log_path.string());

  if (content.empty()) {
    textView_.text =
        [NSString stringWithFormat:@"No recent log data found.\n\nExpected file:\n%@\n\n"
                                   @"Launch a game, then refresh this screen.",
                                   log_path_ns];
  } else {
    NSData* log_data = [NSData dataWithBytes:content.data() length:content.size()];
    NSString* decoded = [[NSString alloc] initWithData:log_data encoding:NSUTF8StringEncoding];
    if (!decoded) {
      decoded = [[NSString alloc] initWithData:log_data encoding:NSISOLatin1StringEncoding];
    }
    if (!decoded) {
      decoded = @"<Unable to decode log data>";
    }
    textView_.text = decoded;
    [textView_ scrollRangeToVisible:NSMakeRange(textView_.text.length, 0)];
  }

  footerLabel_.text =
      [NSString stringWithFormat:@"Showing last %zu KB from %@", kMaxLogBytes / 1024, log_path_ns];
}

@end

@implementation XeniaLandscapeNavigationController

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
  return UIInterfaceOrientationMaskLandscape;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationLandscapeLeft;
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
  return UIInterfaceOrientationMaskLandscape;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationLandscapeLeft;
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
@interface XeniaViewController
    : UIViewController <UIDocumentPickerDelegate>
@property(nonatomic, strong) XeniaMetalView* metalView;
@property(nonatomic, strong) UIView* launcherOverlay;
@property(nonatomic, strong) UIButton* openGameButton;
@property(nonatomic, strong) UIButton* settingsButton;
@property(nonatomic, strong) UILabel* titleLabel;
@property(nonatomic, strong) UILabel* statusLabel;
@property(nonatomic, strong) NSURL* securityScopedURL;
@property(nonatomic, assign) xe::ui::IOSWindowedAppContext* appContext;

// JIT gate overlay.
@property(nonatomic, strong) UIView* jitOverlay;
@property(nonatomic, strong) UIView* jitPulseView;
@property(nonatomic, strong) UIView* jitStatusDot;
@property(nonatomic, strong) UILabel* jitStatusLabel;
@property(nonatomic, strong) NSTimer* jitPollTimer;
@property(nonatomic, assign) BOOL jitAcquired;
@end

@implementation XeniaViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  self.view.backgroundColor = [UIColor blackColor];
  self.jitAcquired = NO;

  // Create the Metal-backed rendering view (full screen, behind everything).
  self.metalView = [[XeniaMetalView alloc] initWithFrame:self.view.bounds];
  self.metalView.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.metalView.contentScaleFactor = [UIScreen mainScreen].scale;
  [self.view addSubview:self.metalView];

  // Create the launcher overlay UI (starts hidden until JIT is acquired).
  [self setupLauncherOverlay];
  self.launcherOverlay.hidden = YES;
  self.launcherOverlay.alpha = 0.0;

  // Create the JIT gate overlay (blocks interaction until JIT is available).
  [self setupJITOverlay];

  // Start polling for JIT.
  [self startJITPoll];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
  xe_request_landscape_orientation(self);
}

// ---------------------------------------------------------------------------
// JIT gate overlay -- shown until JIT is acquired.
// ---------------------------------------------------------------------------
- (void)setupJITOverlay {
  self.jitOverlay = [[UIView alloc] initWithFrame:self.view.bounds];
  self.jitOverlay.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.jitOverlay.backgroundColor = [UIColor colorWithWhite:0.0 alpha:0.95];
  [self.view addSubview:self.jitOverlay];

  // Container for centered content.
  UIView* container = [[UIView alloc] init];
  container.translatesAutoresizingMaskIntoConstraints = NO;
  [self.jitOverlay addSubview:container];

  // Pulsing circle behind the icon.
  self.jitPulseView = [[UIView alloc] init];
  self.jitPulseView.translatesAutoresizingMaskIntoConstraints = NO;
  self.jitPulseView.backgroundColor = [UIColor colorWithRed:0.0 green:0.478 blue:1.0 alpha:0.15];
  self.jitPulseView.layer.cornerRadius = 50;
  [container addSubview:self.jitPulseView];

  // CPU icon.
  UIImageView* iconView = [[UIImageView alloc]
      initWithImage:
          [UIImage systemImageNamed:@"cpu"
                  withConfiguration:[UIImageSymbolConfiguration
                                        configurationWithPointSize:50
                                                            weight:UIImageSymbolWeightMedium]]];
  iconView.tintColor = [UIColor systemBlueColor];
  iconView.translatesAutoresizingMaskIntoConstraints = NO;
  iconView.contentMode = UIViewContentModeCenter;
  [container addSubview:iconView];

  // "Waiting for JIT" title.
  UILabel* jitTitle = [[UILabel alloc] init];
  jitTitle.text = @"Waiting for JIT";
  jitTitle.textColor = [UIColor whiteColor];
  jitTitle.font = [UIFont systemFontOfSize:24 weight:UIFontWeightSemibold];
  jitTitle.textAlignment = NSTextAlignmentCenter;
  jitTitle.translatesAutoresizingMaskIntoConstraints = NO;
  [container addSubview:jitTitle];

  // Subtitle.
  UILabel* jitSubtitle = [[UILabel alloc] init];
  jitSubtitle.text = @"Waiting for Just-In-Time compilation...";
  jitSubtitle.textColor = [UIColor secondaryLabelColor];
  jitSubtitle.font = [UIFont systemFontOfSize:15 weight:UIFontWeightRegular];
  jitSubtitle.textAlignment = NSTextAlignmentCenter;
  jitSubtitle.translatesAutoresizingMaskIntoConstraints = NO;
  [container addSubview:jitSubtitle];

  // Info card background.
  UIView* infoCard = [[UIView alloc] init];
  infoCard.translatesAutoresizingMaskIntoConstraints = NO;
  infoCard.backgroundColor = [UIColor colorWithWhite:0.15 alpha:1.0];
  infoCard.layer.cornerRadius = 12;
  [container addSubview:infoCard];

  // Info row 1: blue info icon + description.
  UIImageView* infoIcon =
      [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"info.circle.fill"]];
  infoIcon.tintColor = [UIColor systemBlueColor];
  infoIcon.translatesAutoresizingMaskIntoConstraints = NO;
  [infoCard addSubview:infoIcon];

  UILabel* infoText = [[UILabel alloc] init];
  infoText.text = @"JIT compilation is required for Xenia to run "
                  @"Xbox 360 games. It dynamically translates and "
                  @"executes code at full speed.";
  infoText.textColor = [UIColor secondaryLabelColor];
  infoText.font = [UIFont systemFontOfSize:13];
  infoText.numberOfLines = 0;
  infoText.translatesAutoresizingMaskIntoConstraints = NO;
  [infoCard addSubview:infoText];

  // Info row 2: green check icon + instructions.
  UIImageView* checkIcon =
      [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"checkmark.circle.fill"]];
  checkIcon.tintColor = [UIColor systemGreenColor];
  checkIcon.translatesAutoresizingMaskIntoConstraints = NO;
  [infoCard addSubview:checkIcon];

  UILabel* checkText = [[UILabel alloc] init];
  checkText.text = @"Enable JIT using StikDebug, SideJITServer, "
                   @"or AltJIT. If running from Xcode, attach the "
                   @"debugger.";
  checkText.textColor = [UIColor secondaryLabelColor];
  checkText.font = [UIFont systemFontOfSize:13];
  checkText.numberOfLines = 0;
  checkText.translatesAutoresizingMaskIntoConstraints = NO;
  [infoCard addSubview:checkText];

  // Layout.
  CGFloat maxCardWidth = 360;
  [NSLayoutConstraint activateConstraints:@[
    // Container centered in overlay.
    [container.centerXAnchor constraintEqualToAnchor:self.jitOverlay.centerXAnchor],
    [container.centerYAnchor constraintEqualToAnchor:self.jitOverlay.centerYAnchor],
    [container.widthAnchor constraintLessThanOrEqualToConstant:maxCardWidth],
    [container.leadingAnchor constraintGreaterThanOrEqualToAnchor:self.jitOverlay.leadingAnchor
                                                         constant:24],

    // Pulse circle.
    [self.jitPulseView.centerXAnchor constraintEqualToAnchor:container.centerXAnchor],
    [self.jitPulseView.topAnchor constraintEqualToAnchor:container.topAnchor],
    [self.jitPulseView.widthAnchor constraintEqualToConstant:100],
    [self.jitPulseView.heightAnchor constraintEqualToConstant:100],

    // Icon centered on pulse.
    [iconView.centerXAnchor constraintEqualToAnchor:self.jitPulseView.centerXAnchor],
    [iconView.centerYAnchor constraintEqualToAnchor:self.jitPulseView.centerYAnchor],

    // Title below icon.
    [jitTitle.topAnchor constraintEqualToAnchor:self.jitPulseView.bottomAnchor constant:20],
    [jitTitle.centerXAnchor constraintEqualToAnchor:container.centerXAnchor],

    // Subtitle below title.
    [jitSubtitle.topAnchor constraintEqualToAnchor:jitTitle.bottomAnchor constant:8],
    [jitSubtitle.centerXAnchor constraintEqualToAnchor:container.centerXAnchor],

    // Info card below subtitle.
    [infoCard.topAnchor constraintEqualToAnchor:jitSubtitle.bottomAnchor constant:24],
    [infoCard.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
    [infoCard.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
    [infoCard.bottomAnchor constraintEqualToAnchor:container.bottomAnchor],

    // Info row 1.
    [infoIcon.leadingAnchor constraintEqualToAnchor:infoCard.leadingAnchor constant:14],
    [infoIcon.topAnchor constraintEqualToAnchor:infoCard.topAnchor constant:14],
    [infoIcon.widthAnchor constraintEqualToConstant:18],
    [infoIcon.heightAnchor constraintEqualToConstant:18],

    [infoText.leadingAnchor constraintEqualToAnchor:infoIcon.trailingAnchor constant:10],
    [infoText.trailingAnchor constraintEqualToAnchor:infoCard.trailingAnchor constant:-14],
    [infoText.topAnchor constraintEqualToAnchor:infoIcon.topAnchor constant:-2],

    // Info row 2.
    [checkIcon.leadingAnchor constraintEqualToAnchor:infoCard.leadingAnchor constant:14],
    [checkIcon.topAnchor constraintEqualToAnchor:infoText.bottomAnchor constant:14],
    [checkIcon.widthAnchor constraintEqualToConstant:18],
    [checkIcon.heightAnchor constraintEqualToConstant:18],

    [checkText.leadingAnchor constraintEqualToAnchor:checkIcon.trailingAnchor constant:10],
    [checkText.trailingAnchor constraintEqualToAnchor:infoCard.trailingAnchor constant:-14],
    [checkText.topAnchor constraintEqualToAnchor:checkIcon.topAnchor constant:-2],
    [checkText.bottomAnchor constraintEqualToAnchor:infoCard.bottomAnchor constant:-14],
  ]];

  // Start the pulsing animation.
  [self startPulseAnimation];
}

- (void)startPulseAnimation {
  self.jitPulseView.alpha = 1.0;
  self.jitPulseView.transform = CGAffineTransformIdentity;

  [UIView animateWithDuration:1.5
                        delay:0.0
                      options:UIViewAnimationOptionCurveEaseInOut | UIViewAnimationOptionRepeat |
                              UIViewAnimationOptionAllowUserInteraction
                   animations:^{
                     self.jitPulseView.transform = CGAffineTransformMakeScale(1.3, 1.3);
                     self.jitPulseView.alpha = 0.0;
                   }
                   completion:nil];
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

  // Update the JIT status indicator on the launcher.
  [self updateJITStatusIndicator];

  // Fade out JIT overlay, reveal launcher.
  self.launcherOverlay.hidden = NO;
  [UIView animateWithDuration:0.4
      delay:0.0
      options:UIViewAnimationOptionCurveEaseOut
      animations:^{
        self.jitOverlay.alpha = 0.0;
        self.launcherOverlay.alpha = 1.0;
      }
      completion:^(BOOL finished) {
        [self.jitOverlay removeFromSuperview];
        self.jitOverlay = nil;
      }];
}

// ---------------------------------------------------------------------------
// Launcher overlay with Open Game button.
// ---------------------------------------------------------------------------
- (void)setupLauncherOverlay {
  self.launcherOverlay =
      [[UIView alloc] initWithFrame:self.view.bounds];
  self.launcherOverlay.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.launcherOverlay.backgroundColor =
      [UIColor colorWithWhite:0.0 alpha:0.85];
  [self.view addSubview:self.launcherOverlay];

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
  self.jitStatusLabel.text = @"JIT Not Acquired";
  self.jitStatusLabel.textColor = [UIColor systemRedColor];
  self.jitStatusLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightMedium];
  self.jitStatusLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [statusContainer addSubview:self.jitStatusLabel];

  // Title label.
  self.titleLabel = [[UILabel alloc] init];
  self.titleLabel.text = @"Xenia";
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

  // Open Game button.
  UIButtonConfiguration* config =
      [UIButtonConfiguration filledButtonConfiguration];
  config.title = @"Open Game";
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

  // Status label (for showing loading state).
  self.statusLabel = [[UILabel alloc] init];
  self.statusLabel.text = @"";
  self.statusLabel.textColor = [UIColor lightGrayColor];
  self.statusLabel.font =
      [UIFont systemFontOfSize:14 weight:UIFontWeightRegular];
  self.statusLabel.textAlignment = NSTextAlignmentCenter;
  self.statusLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:self.statusLabel];

  // Layout constraints.
  [NSLayoutConstraint activateConstraints:@[
    // JIT status indicator at top center.
    [statusContainer.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [statusContainer.topAnchor
        constraintEqualToAnchor:self.launcherOverlay.safeAreaLayoutGuide.topAnchor
                       constant:16],

    [self.jitStatusDot.leadingAnchor constraintEqualToAnchor:statusContainer.leadingAnchor],
    [self.jitStatusDot.centerYAnchor constraintEqualToAnchor:statusContainer.centerYAnchor],
    [self.jitStatusDot.widthAnchor constraintEqualToConstant:10],
    [self.jitStatusDot.heightAnchor constraintEqualToConstant:10],

    [self.jitStatusLabel.leadingAnchor constraintEqualToAnchor:self.jitStatusDot.trailingAnchor
                                                      constant:6],
    [self.jitStatusLabel.trailingAnchor constraintEqualToAnchor:statusContainer.trailingAnchor],
    [self.jitStatusLabel.centerYAnchor constraintEqualToAnchor:statusContainer.centerYAnchor],

    [self.settingsButton.trailingAnchor
        constraintEqualToAnchor:self.launcherOverlay.safeAreaLayoutGuide.trailingAnchor
                       constant:-16],
    [self.settingsButton.topAnchor
        constraintEqualToAnchor:self.launcherOverlay.safeAreaLayoutGuide.topAnchor
                       constant:12],

    // Title and subtitle.
    [self.titleLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.titleLabel.bottomAnchor constraintEqualToAnchor:subtitleLabel.topAnchor constant:-4],

    [subtitleLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [subtitleLabel.bottomAnchor constraintEqualToAnchor:self.openGameButton.topAnchor constant:-32],

    [self.openGameButton.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.openGameButton.centerYAnchor constraintEqualToAnchor:self.launcherOverlay.centerYAnchor
                                                      constant:20],

    [self.statusLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.statusLabel.topAnchor constraintEqualToAnchor:self.openGameButton.bottomAnchor
                                               constant:20],
  ]];
}

- (void)updateJITStatusIndicator {
  if (self.jitAcquired) {
    self.jitStatusDot.backgroundColor = [UIColor systemGreenColor];
    self.jitStatusLabel.text = @"JIT Enabled";
    self.jitStatusLabel.textColor = [UIColor systemGreenColor];
  } else {
    self.jitStatusDot.backgroundColor = [UIColor systemRedColor];
    self.jitStatusLabel.text = @"JIT Not Acquired";
    self.jitStatusLabel.textColor = [UIColor systemRedColor];
  }
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
    [UTType typeWithFilenameExtension:@"zar"],
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
  xe_request_landscape_orientation(self);
  [self presentViewController:nav animated:YES completion:nil];
}

#pragma mark - UIDocumentPickerDelegate

- (void)documentPicker:(UIDocumentPickerViewController*)controller
    didPickDocumentsAtURLs:(NSArray<NSURL*>*)urls {
  if (urls.count == 0) return;

  NSURL* url = urls[0];

  // Stop any previous security-scoped access before starting a new one.
  if (self.securityScopedURL) {
    [self.securityScopedURL stopAccessingSecurityScopedResource];
    self.securityScopedURL = nil;
  }

  // Start security-scoped access for files outside the sandbox.
  BOOL accessGranted = [url startAccessingSecurityScopedResource];
  if (accessGranted) {
    self.securityScopedURL = url;
  }

  NSString* path = url.path;
  XELOGI("iOS: User selected game file: {} (security-scoped: {})", [path UTF8String],
         accessGranted ? "yes" : "no");

  // Save a bookmark for potential future relaunch.
  NSError* bookmarkError = nil;
  NSData* bookmark = [url bookmarkDataWithOptions:0
                   includingResourceValuesForKeys:nil
                                    relativeToURL:nil
                                            error:&bookmarkError];
  if (bookmark) {
    [[NSUserDefaults standardUserDefaults] setObject:bookmark forKey:@"lastGameBookmark"];
  }

  // Update UI to show loading state.
  self.statusLabel.text =
      [NSString stringWithFormat:@"Loading: %@", url.lastPathComponent];
  self.openGameButton.enabled = NO;

  // Hide the launcher overlay with animation.
  [UIView animateWithDuration:0.3
      animations:^{
        self.launcherOverlay.alpha = 0.0;
      }
      completion:^(BOOL finished) {
        self.launcherOverlay.hidden = YES;
      }];

  // Launch the game through the app context callback.
  if (self.appContext) {
    std::string game_path = std::string([path UTF8String]);
    self.appContext->LaunchGame(game_path);
  }
}

- (void)documentPickerWasCancelled:
    (UIDocumentPickerViewController*)controller {
  XELOGI("iOS: Document picker cancelled");
}

#pragma mark - Status bar / home indicator

- (BOOL)prefersStatusBarHidden {
  return YES;
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
  return UIInterfaceOrientationMaskLandscape;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
  return UIInterfaceOrientationLandscapeLeft;
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
  self.launcherOverlay.hidden = NO;
  self.openGameButton.enabled = YES;
  self.statusLabel.text = @"";
  [self updateJITStatusIndicator];
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
  xe_request_landscape_orientation(vc);

  // Force layout so the Metal view is created.
  [vc.view layoutIfNeeded];

  // Store the Metal view and view controller in the app context for
  // iOSWindow to use.
  app_context_->set_metal_view(vc.metalView);
  app_context_->set_view_controller(vc);
  vc.appContext = app_context_.get();

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

  XELOGI("iOS: Application launched successfully");
  return YES;
}

- (UIInterfaceOrientationMask)application:(UIApplication*)application
    supportedInterfaceOrientationsForWindow:(UIWindow*)window {
  return UIInterfaceOrientationMaskLandscape;
}

- (void)applicationWillTerminate:(UIApplication*)application {
  XELOGI("iOS lifecycle: applicationWillTerminate");
  // Release security-scoped file access.
  XeniaViewController* vc = (XeniaViewController*)self.window.rootViewController;
  if (vc.securityScopedURL) {
    [vc.securityScopedURL stopAccessingSecurityScopedResource];
    vc.securityScopedURL = nil;
  }

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
