/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/file_picker.h"

#import <UIKit/UIKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include "xenia/base/logging.h"

// Delegate for handling document picker results.
@interface XeniaDocumentPickerDelegate
    : NSObject <UIDocumentPickerDelegate>
@property(nonatomic, assign) bool completed;
@property(nonatomic, assign) bool cancelled;
@property(nonatomic, strong) NSArray<NSURL*>* selectedURLs;
@end

@implementation XeniaDocumentPickerDelegate

- (void)documentPicker:(UIDocumentPickerViewController*)controller
    didPickDocumentsAtURLs:(NSArray<NSURL*>*)urls {
  self.selectedURLs = urls;
  self.completed = true;
  self.cancelled = false;
}

- (void)documentPickerWasCancelled:
    (UIDocumentPickerViewController*)controller {
  self.completed = true;
  self.cancelled = true;
}

@end

namespace xe {
namespace ui {

class IOSFilePicker : public FilePicker {
 public:
  IOSFilePicker() = default;
  ~IOSFilePicker() override = default;

  bool Show(Window* parent_window) override;
};

std::unique_ptr<FilePicker> FilePicker::Create() {
  return std::make_unique<IOSFilePicker>();
}

bool IOSFilePicker::Show(Window* parent_window) {
  @autoreleasepool {
    // Build the list of content types from extensions.
    NSMutableArray<UTType*>* content_types = [NSMutableArray array];

    // Always include common game file types.
    [content_types addObject:[UTType typeWithFilenameExtension:@"iso"]];
    [content_types addObject:[UTType typeWithFilenameExtension:@"xex"]];
    [content_types addObject:[UTType typeWithFilenameExtension:@"zar"]];
    // STFS containers often have no extension, so include generic data.
    [content_types addObject:UTTypeData];

    // Add any extensions the caller specified.
    for (const auto& ext_pair : extensions_) {
      NSString* ext = [NSString
          stringWithUTF8String:ext_pair.second.c_str()];
      // Extensions may contain semicolons for multiple patterns.
      NSArray<NSString*>* parts = [ext componentsSeparatedByString:@";"];
      for (NSString* part in parts) {
        NSString* clean = [part stringByTrimmingCharactersInSet:
            [NSCharacterSet characterSetWithCharactersInString:@"*. "]];
        if (clean.length > 0) {
          UTType* type = [UTType typeWithFilenameExtension:clean];
          if (type) {
            [content_types addObject:type];
          }
        }
      }
    }

    UIDocumentPickerViewController* picker =
        [[UIDocumentPickerViewController alloc]
            initForOpeningContentTypes:content_types];
    picker.allowsMultipleSelection = multi_selection_ ? YES : NO;

    if (!initial_directory_.empty()) {
      NSString* dir_path = [NSString
          stringWithUTF8String:initial_directory_.c_str()];
      NSURL* dir_url = [NSURL fileURLWithPath:dir_path isDirectory:YES];
      picker.directoryURL = dir_url;
    }

    XeniaDocumentPickerDelegate* delegate =
        [[XeniaDocumentPickerDelegate alloc] init];
    delegate.completed = false;
    delegate.cancelled = false;
    picker.delegate = delegate;

    // Present the picker from the root view controller.
    UIViewController* presenting_vc = nil;
    UIWindowScene* scene =
        (UIWindowScene*)[[UIApplication sharedApplication]
                .connectedScenes anyObject];
    if (scene) {
      presenting_vc = scene.keyWindow.rootViewController;
    }
    if (!presenting_vc) {
      XELOGE("IOSFilePicker: No presenting view controller available");
      return false;
    }

    // Present modally and run the run loop until the delegate fires.
    [presenting_vc presentViewController:picker animated:YES completion:nil];

    // Spin the main run loop until the picker completes.
    // This matches the synchronous Show() API contract.
    while (!delegate.completed) {
      [[NSRunLoop mainRunLoop] runMode:NSDefaultRunLoopMode
                            beforeDate:[NSDate dateWithTimeIntervalSinceNow:
                                                   0.05]];
    }

    if (delegate.cancelled || !delegate.selectedURLs ||
        delegate.selectedURLs.count == 0) {
      return false;
    }

    selected_files_.clear();
    for (NSURL* url in delegate.selectedURLs) {
      // Start security-scoped access for files outside the app sandbox.
      [url startAccessingSecurityScopedResource];
      std::string path_str = std::string([url.path UTF8String]);
      selected_files_.push_back(std::filesystem::path(path_str));
    }

    return !selected_files_.empty();
  }
}

}  // namespace ui
}  // namespace xe
