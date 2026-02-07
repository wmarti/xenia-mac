/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <dlfcn.h>
#include <stdlib.h>
#include <sys/resource.h>

#include "xenia/base/assert.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/base/string.h"
#include "xenia/base/system.h"

#if !XE_PLATFORM_IOS
#include <alloca.h>

#include <cstring>

// Use headers in third party to not depend on system sdl headers for building
#include "third_party/SDL2/include/SDL.h"
#endif  // !XE_PLATFORM_IOS

namespace xe {

void LaunchWebBrowser(const std::string_view url) {
#if XE_PLATFORM_IOS
  // TODO(wmarti): Implement via UIApplication openURL.
  XELOGW("LaunchWebBrowser not yet implemented on iOS: {}", url);
#else
  auto cmd = std::string("open ");
  cmd.append(url);
  system(cmd.c_str());
#endif
}

void LaunchFileExplorer(const std::filesystem::path& path) { assert_always(); }

void ShowSimpleMessageBox(SimpleMessageBoxType type, std::string_view message) {
#if XE_PLATFORM_IOS
  // TODO(wmarti): Implement via UIAlertController.
  XELOGW("ShowSimpleMessageBox (iOS): {}", message);
#else
  // Try multiple library names for cross-platform compatibility
  void* libsdl2 = dlopen("libSDL2.dylib", RTLD_LAZY | RTLD_LOCAL);
  if (!libsdl2) {
    libsdl2 = dlopen("libSDL2-2.0.0.dylib", RTLD_LAZY | RTLD_LOCAL);
  }
  if (!libsdl2) {
    libsdl2 = dlopen("/opt/homebrew/lib/libSDL2.dylib", RTLD_LAZY | RTLD_LOCAL);
  }
  // Fallback: just log the message if SDL2 is not available
  if (!libsdl2) {
    XELOGE("ShowSimpleMessageBox (SDL2 not available): {}", message);
    return;
  }
  if (libsdl2) {
    auto* pSDL_ShowSimpleMessageBox =
        reinterpret_cast<decltype(SDL_ShowSimpleMessageBox)*>(
            dlsym(libsdl2, "SDL_ShowSimpleMessageBox"));
    assert_not_null(pSDL_ShowSimpleMessageBox);
    if (pSDL_ShowSimpleMessageBox) {
      Uint32 flags;
      const char* title;
      char* message_copy = reinterpret_cast<char*>(alloca(message.size() + 1));
      std::memcpy(message_copy, message.data(), message.size());
      message_copy[message.size()] = '\0';

      switch (type) {
        default:
        case SimpleMessageBoxType::Help:
          title = "Xenia Help";
          flags = SDL_MESSAGEBOX_INFORMATION;
          break;
        case SimpleMessageBoxType::Warning:
          title = "Xenia Warning";
          flags = SDL_MESSAGEBOX_WARNING;
          break;
        case SimpleMessageBoxType::Error:
          title = "Xenia Error";
          flags = SDL_MESSAGEBOX_ERROR;
          break;
      }
      pSDL_ShowSimpleMessageBox(flags, title, message_copy, NULL);
    }
    dlclose(libsdl2);
  }
#endif  // XE_PLATFORM_IOS
}

bool SetProcessPriorityClass(const uint32_t priority_class) {
  int nice_value = 0;
  switch (priority_class) {
    case 0:
      nice_value = 0;
      break;
    case 1:
      nice_value = -5;
      break;
    case 2:
      nice_value = -10;
      break;
    case 3:
      nice_value = -20;
      break;
    default:
      return false;
  }

  return setpriority(PRIO_PROCESS, 0, nice_value) == 0;
}

bool IsUseNexusForGameBarEnabled() { return false; }

}  // namespace xe
