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

bool iOSUIViewSurface::GetSizeImpl(uint32_t& width_out,
                                   uint32_t& height_out) const {
  if (!view_) {
    width_out = 0;
    height_out = 0;
    return false;
  }

  // Return size in points (logical size), not pixels.
  // MetalPresenter multiplies by contentScaleFactor (backing_scale) to get
  // the drawable size in pixels. Returning pixels here would cause
  // double-scaling (points * scale * scale), leading to massive drawable
  // sizes, allocation failures, or extreme slowdown on high-DPI iOS devices.
  CGRect bounds = [view_ bounds];
  width_out = static_cast<uint32_t>(bounds.size.width);
  height_out = static_cast<uint32_t>(bounds.size.height);

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
    return static_cast<CAMetalLayer*>(layer);
  }

  // Create and set a new CAMetalLayer.
  CAMetalLayer* metalLayer = [CAMetalLayer layer];
  [view_ setNeedsLayout];
  [view_.layer addSublayer:metalLayer];
  metalLayer.frame = view_.bounds;
  metalLayer.contentsScale = [UIScreen mainScreen].scale;
  metalLayer.framebufferOnly = YES;
  metalLayer.opaque = YES;

  return metalLayer;
}

}  // namespace ui
}  // namespace xe
