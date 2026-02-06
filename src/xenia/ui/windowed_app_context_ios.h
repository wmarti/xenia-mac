/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_WINDOWED_APP_CONTEXT_IOS_H_
#define XENIA_UI_WINDOWED_APP_CONTEXT_IOS_H_

#include "xenia/ui/windowed_app_context.h"

namespace xe {
namespace ui {

class IOSWindowedAppContext final : public WindowedAppContext {
 public:
  IOSWindowedAppContext();
  ~IOSWindowedAppContext();

  void NotifyUILoopOfPendingFunctions() override;
  void PlatformQuitFromUIThread() override;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOWED_APP_CONTEXT_IOS_H_
