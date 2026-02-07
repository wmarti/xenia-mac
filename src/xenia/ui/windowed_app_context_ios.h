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

#include <functional>

#include "xenia/ui/windowed_app_context.h"

#ifdef __OBJC__
@class UIView;
@class UIViewController;
#else
typedef struct objc_object UIView;
typedef struct objc_object UIViewController;
#endif

namespace xe {
namespace ui {

class IOSWindowedAppContext final : public WindowedAppContext {
 public:
  IOSWindowedAppContext();
  ~IOSWindowedAppContext();

  void NotifyUILoopOfPendingFunctions() override;
  void PlatformQuitFromUIThread() override;

  // The Metal-backed rendering view, set by the app delegate after UIKit
  // hierarchy creation.
  UIView* metal_view() const { return metal_view_; }
  void set_metal_view(UIView* view) { metal_view_ = view; }

  UIViewController* view_controller() const { return view_controller_; }
  void set_view_controller(UIViewController* vc) { view_controller_ = vc; }

  // Callback invoked when the user selects a game file to launch.
  using GameLaunchCallback = std::function<void(const std::string&)>;
  void set_game_launch_callback(GameLaunchCallback callback) {
    game_launch_callback_ = std::move(callback);
  }
  void LaunchGame(const std::string& path) {
    if (game_launch_callback_) {
      game_launch_callback_(path);
    }
  }

 private:
  UIView* metal_view_ = nullptr;
  UIViewController* view_controller_ = nullptr;
  GameLaunchCallback game_launch_callback_;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOWED_APP_CONTEXT_IOS_H_
