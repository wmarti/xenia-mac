/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/surface_ios.h"

#import <UIKit/UIKit.h>
#import <QuartzCore/QuartzCore.h>

namespace xe {
namespace ui {

namespace {
constexpr uint32_t kForcedLogicalWidth = 1280;
constexpr uint32_t kForcedLogicalHeight = 720;
}  // namespace

bool iOSUIViewSurface::GetSizeImpl(uint32_t& width_out,
                                   uint32_t& height_out) const {
  if (!view_) {
    width_out = 0;
    height_out = 0;
    return false;
  }

  // Force fixed logical rendering resolution on iOS.
  width_out = kForcedLogicalWidth;
  height_out = kForcedLogicalHeight;

  return true;
}

CAMetalLayer* iOSUIViewSurface::GetOrCreateMetalLayer() const {
  if (!view_) {
    return nullptr;
  }

  // On iOS, UIView's layer can be set to a CAMetalLayer via layerClass
  // or by replacing the layer directly.
  CALayer* layer = [view_ layer];
  if ([layer isKindOfClass:[CAMetalLayer class]]) {
    CAMetalLayer* metal_layer = static_cast<CAMetalLayer*>(layer);
    // Keep fixed-size drawables from stretching on non-16:9 displays.
    metal_layer.contentsGravity = kCAGravityResizeAspect;
    return metal_layer;
  }

  // Create and set a new CAMetalLayer.
  CAMetalLayer* metalLayer = [CAMetalLayer layer];
  [view_ setNeedsLayout];
  [view_.layer addSublayer:metalLayer];
  metalLayer.frame = view_.bounds;
  metalLayer.contentsScale = [UIScreen mainScreen].scale;
  metalLayer.contentsGravity = kCAGravityResizeAspect;
  metalLayer.framebufferOnly = YES;
  metalLayer.opaque = YES;

  return metalLayer;
}

}  // namespace ui
}  // namespace xe
