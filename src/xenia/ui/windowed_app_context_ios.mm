/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/windowed_app_context_ios.h"

#import <UIKit/UIKit.h>
#import <dispatch/dispatch.h>

namespace xe {
namespace ui {

IOSWindowedAppContext::IOSWindowedAppContext() = default;

IOSWindowedAppContext::~IOSWindowedAppContext() = default;

void IOSWindowedAppContext::NotifyUILoopOfPendingFunctions() {
  // Use GCD to schedule pending function execution on the main thread.
  dispatch_async(dispatch_get_main_queue(), ^{
    ExecutePendingFunctionsFromUIThread();
  });
}

void IOSWindowedAppContext::PlatformQuitFromUIThread() {
  // iOS apps don't self-terminate; the OS manages the lifecycle.
  // Suspending to background is the closest equivalent.
}

}  // namespace ui
}  // namespace xe
