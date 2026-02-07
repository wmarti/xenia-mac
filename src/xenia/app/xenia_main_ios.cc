/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <memory>

#include "xenia/base/logging.h"
#include "xenia/ui/windowed_app.h"
#include "xenia/ui/windowed_app_context.h"

namespace xe {
namespace app {

// Minimal iOS app stub. The full emulator setup (graphics, audio, input)
// will be added as iOS platform support matures.
class EmulatorAppIOS final : public xe::ui::WindowedApp {
 public:
  static std::unique_ptr<xe::ui::WindowedApp> Create(
      xe::ui::WindowedAppContext& app_context) {
    return std::unique_ptr<xe::ui::WindowedApp>(
        new EmulatorAppIOS(app_context));
  }

  bool OnInitialize() override {
    XELOGI("iOS: EmulatorAppIOS initialized");
    return true;
  }

 private:
  explicit EmulatorAppIOS(xe::ui::WindowedAppContext& app_context)
      : xe::ui::WindowedApp(app_context, "xenia") {}
};

}  // namespace app
}  // namespace xe

XE_DEFINE_WINDOWED_APP(xenia, xe::app::EmulatorAppIOS::Create);
