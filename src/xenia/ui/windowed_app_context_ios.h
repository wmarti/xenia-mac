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

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

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

struct IOSProfileSummary {
  uint64_t xuid = 0;
  std::string gamertag;
  bool signed_in = false;
  uint8_t signed_in_slot = 0xFF;
};

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

  using ProfilesListCallback = std::function<std::vector<IOSProfileSummary>()>;
  using ProfileCreateCallback = std::function<uint64_t(const std::string&)>;
  using ProfileSignInCallback = std::function<bool(uint64_t)>;
  using GameTerminateCallback = std::function<bool()>;
  using GameExitedCallback = std::function<void()>;
  using ProfileServicesReadyCallback = std::function<void()>;
  using SignInUIPromptCallback = std::function<bool(uint32_t, uint32_t)>;
  using MessageBoxPromptCallback =
      std::function<bool(const std::string&, const std::string&, const std::vector<std::string>&,
                         uint32_t, uint32_t*)>;
  using KeyboardPromptCallback = std::function<bool(const std::string&, const std::string&,
                                                    const std::string&, std::string*, bool*)>;

  void set_profiles_list_callback(ProfilesListCallback callback) {
    profiles_list_callback_ = std::move(callback);
  }
  std::vector<IOSProfileSummary> ListProfiles() const {
    if (!profiles_list_callback_) {
      return {};
    }
    return profiles_list_callback_();
  }

  void set_profile_create_callback(ProfileCreateCallback callback) {
    profile_create_callback_ = std::move(callback);
  }
  uint64_t CreateProfile(const std::string& gamertag) const {
    if (!profile_create_callback_) {
      return 0;
    }
    return profile_create_callback_(gamertag);
  }

  void set_profile_sign_in_callback(ProfileSignInCallback callback) {
    profile_sign_in_callback_ = std::move(callback);
  }
  bool SignInProfile(uint64_t xuid) const {
    if (!profile_sign_in_callback_) {
      return false;
    }
    return profile_sign_in_callback_(xuid);
  }

  void set_profile_services_ready_callback(ProfileServicesReadyCallback callback) {
    profile_services_ready_callback_ = std::move(callback);
  }
  void NotifyProfileServicesReady() const {
    if (profile_services_ready_callback_) {
      profile_services_ready_callback_();
    }
  }

  void set_game_terminate_callback(GameTerminateCallback callback) {
    game_terminate_callback_ = std::move(callback);
  }
  bool TerminateCurrentGame() const {
    if (!game_terminate_callback_) {
      return false;
    }
    return game_terminate_callback_();
  }

  void set_game_exited_callback(GameExitedCallback callback) {
    game_exited_callback_ = std::move(callback);
  }
  void NotifyGameExited() {
    if (game_exited_callback_) {
      game_exited_callback_();
    }
  }

  void set_signin_ui_prompt_callback(SignInUIPromptCallback callback) {
    signin_ui_prompt_callback_ = std::move(callback);
  }
  bool PromptSignInUI(uint32_t user_index, uint32_t users_needed) const {
    if (!signin_ui_prompt_callback_) {
      return false;
    }
    return signin_ui_prompt_callback_(user_index, users_needed);
  }

  void set_message_box_prompt_callback(MessageBoxPromptCallback callback) {
    message_box_prompt_callback_ = std::move(callback);
  }
  bool PromptMessageBoxUI(const std::string& title, const std::string& text,
                          const std::vector<std::string>& buttons, uint32_t default_button,
                          uint32_t* selected_button_out) const;

  void set_keyboard_prompt_callback(KeyboardPromptCallback callback) {
    keyboard_prompt_callback_ = std::move(callback);
  }
  bool PromptKeyboardUI(const std::string& title, const std::string& description,
                        const std::string& default_text, std::string* text_out,
                        bool* cancelled_out) const {
    if (!keyboard_prompt_callback_) {
      return false;
    }
    return keyboard_prompt_callback_(title, description, default_text, text_out, cancelled_out);
  }

  // Callback invoked when the view layout changes (rotation, resize, etc.).
  using LayoutChangedCallback = std::function<void()>;
  void set_layout_changed_callback(LayoutChangedCallback callback) {
    layout_changed_callback_ = std::move(callback);
  }
  void NotifyLayoutChanged() {
    if (layout_changed_callback_) {
      layout_changed_callback_();
    }
  }

 private:
  UIView* metal_view_ = nullptr;
  UIViewController* view_controller_ = nullptr;
  GameLaunchCallback game_launch_callback_;
  ProfilesListCallback profiles_list_callback_;
  ProfileCreateCallback profile_create_callback_;
  ProfileSignInCallback profile_sign_in_callback_;
  GameTerminateCallback game_terminate_callback_;
  GameExitedCallback game_exited_callback_;
  SignInUIPromptCallback signin_ui_prompt_callback_;
  ProfileServicesReadyCallback profile_services_ready_callback_;
  MessageBoxPromptCallback message_box_prompt_callback_;
  KeyboardPromptCallback keyboard_prompt_callback_;
  LayoutChangedCallback layout_changed_callback_;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOWED_APP_CONTEXT_IOS_H_
