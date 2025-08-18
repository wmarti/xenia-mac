/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_RENDERDOC_API_H_
#define XENIA_UI_RENDERDOC_API_H_

#include "xenia/base/platform.h"

// RenderDoc is not supported on macOS
#if !XE_PLATFORM_MAC
#include "third_party/renderdoc/renderdoc_app.h"
#endif

namespace xe {
namespace ui {

class RenderdocApi {
 public:
  RenderdocApi() = default;
  RenderdocApi(const RenderdocApi& renderdoc_api) = delete;
  RenderdocApi& operator=(const RenderdocApi& renderdoc_api) = delete;
  ~RenderdocApi() { Shutdown(); }

  bool Initialize();
  void Shutdown();

#if !XE_PLATFORM_MAC
  // nullptr if not attached.
  const RENDERDOC_API_1_0_0* api_1_0_0() const { return api_1_0_0_; }
#else
  // Stub for macOS - always returns nullptr
  const void* api_1_0_0() const { return nullptr; }
#endif

 private:
  void* library_ = nullptr;
#if !XE_PLATFORM_MAC
  const RENDERDOC_API_1_0_0* api_1_0_0_ = nullptr;
#else
  const void* api_1_0_0_ = nullptr;
#endif
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_RENDERDOC_API_H_