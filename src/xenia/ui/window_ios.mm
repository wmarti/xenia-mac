/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/window_ios.h"

#import <UIKit/UIKit.h>
#import <QuartzCore/QuartzCore.h>

#include "xenia/base/logging.h"
#include "xenia/ui/surface_ios.h"
#include "xenia/ui/windowed_app_context_ios.h"

namespace xe {
namespace ui {

std::unique_ptr<Window> Window::Create(WindowedAppContext& app_context,
                                       const std::string_view title,
                                       uint32_t desired_logical_width,
                                       uint32_t desired_logical_height,
                                       bool is_game_process) {
  auto window = std::make_unique<iOSWindow>(app_context, title,
                                            desired_logical_width,
                                            desired_logical_height);
  return window;
}

iOSWindow::iOSWindow(WindowedAppContext& app_context,
                      const std::string_view title,
                      uint32_t desired_logical_width,
                      uint32_t desired_logical_height)
    : Window(app_context, title, desired_logical_width,
             desired_logical_height) {}

iOSWindow::~iOSWindow() {
  EnterDestructor();
  TeardownDisplayLink();
}

void iOSWindow::SetNativeView(UIView* view,
                               UIViewController* view_controller) {
  view_ = view;
  view_controller_ = view_controller;
}

void iOSWindow::HandleSizeChange() {
  if (!view_) return;

  CGRect bounds = [view_ bounds];
  CGFloat scale = [view_ contentScaleFactor];
  uint32_t width = static_cast<uint32_t>(bounds.size.width * scale);
  uint32_t height = static_cast<uint32_t>(bounds.size.height * scale);

  WindowDestructionReceiver destruction_receiver(this);
  OnActualSizeUpdate(width, height, destruction_receiver);
}

bool iOSWindow::OpenImpl() {
  auto& ios_context =
      static_cast<IOSWindowedAppContext&>(app_context());
  view_ = ios_context.metal_view();
  view_controller_ = ios_context.view_controller();

  if (!view_) {
    XELOGE("iOSWindow::OpenImpl: No Metal view available from app context");
    return false;
  }

  // On iOS, the window is always effectively fullscreen.
  OnDesiredFullscreenUpdate(true);

  // Report initial monitor info.
  {
    MonitorUpdateEvent monitor_event(this, true, true);
    OnMonitorUpdate(monitor_event);
  }

  // Report initial size.
  CGRect bounds = [view_ bounds];
  CGFloat scale = [view_ contentScaleFactor];
  uint32_t width = static_cast<uint32_t>(bounds.size.width * scale);
  uint32_t height = static_cast<uint32_t>(bounds.size.height * scale);
  {
    WindowDestructionReceiver destruction_receiver(this);
    OnActualSizeUpdate(width, height, destruction_receiver);
    if (destruction_receiver.IsWindowDestroyed()) return true;
  }

  // Report initial focus (iOS app is always in focus when visible).
  {
    WindowDestructionReceiver destruction_receiver(this);
    OnFocusUpdate(true, destruction_receiver);
    if (destruction_receiver.IsWindowDestroyed()) return true;
  }

  SetupDisplayLink();

  XELOGI("iOSWindow: Opened ({}x{} @ {:.0f}x scale)", width, height, scale);
  return true;
}

void iOSWindow::RequestCloseImpl() {
  // iOS apps don't close windows in the traditional sense.
  // Signal the close lifecycle for proper cleanup.
  WindowDestructionReceiver destruction_receiver(this);
  OnBeforeClose(destruction_receiver);
  if (destruction_receiver.IsWindowDestroyed()) return;
  TeardownDisplayLink();
  OnAfterClose();
}

uint32_t iOSWindow::GetLatestDpiImpl() const {
  if (view_) {
    // iOS points are 1/163 of an inch at 1x scale.
    // Standard DPI: 163 * scale factor.
    CGFloat scale = [view_ contentScaleFactor];
    return static_cast<uint32_t>(163.0 * scale);
  }
  // Default: 2x Retina = 326 DPI.
  return 326;
}

void iOSWindow::ApplyNewFullscreen() {
  // iOS is always fullscreen. Update status bar/home indicator visibility.
  if (view_controller_) {
    [view_controller_ setNeedsStatusBarAppearanceUpdate];
    [view_controller_ setNeedsUpdateOfHomeIndicatorAutoHidden];
  }
}

std::unique_ptr<Surface> iOSWindow::CreateSurfaceImpl(
    Surface::TypeFlags allowed_types) {
  if (!view_) return nullptr;
  if (allowed_types & (1 << Surface::kTypeIndex_iOSUIView)) {
    return std::make_unique<iOSUIViewSurface>(view_);
  }
  return nullptr;
}

void iOSWindow::RequestPaintImpl() {
  // Painting is driven by the display link, but we can request an
  // immediate redraw by triggering paint on the next main loop iteration.
  dispatch_async(dispatch_get_main_queue(), ^{
    OnPaint();
  });
}

}  // namespace ui
}  // namespace xe

// Helper Objective-C class for display link callback.
// Must be at global scope (ObjC declarations cannot appear inside C++
// namespaces).
@interface XeniaDisplayLinkTarget : NSObject {
  xe::ui::iOSWindow* _window;
}
- (instancetype)initWithWindow:(xe::ui::iOSWindow*)window;
- (void)displayLinkFired:(CADisplayLink*)link;
@end

@implementation XeniaDisplayLinkTarget
- (instancetype)initWithWindow:(xe::ui::iOSWindow*)window {
  if (self = [super init]) {
    _window = window;
  }
  return self;
}
- (void)displayLinkFired:(CADisplayLink*)link {
  if (_window) {
    _window->TriggerPaint();
  }
}
@end

static XeniaDisplayLinkTarget* g_display_link_target = nil;

namespace xe {
namespace ui {

void iOSWindow::SetupDisplayLink() {
  if (display_link_) return;

  g_display_link_target =
      [[XeniaDisplayLinkTarget alloc] initWithWindow:this];
  display_link_ = [CADisplayLink
      displayLinkWithTarget:g_display_link_target
                   selector:@selector(displayLinkFired:)];
  // Prefer 60 FPS but allow the system to adjust.
  display_link_.preferredFrameRateRange =
      CAFrameRateRangeMake(30.0, 120.0, 60.0);
  [display_link_ addToRunLoop:[NSRunLoop mainRunLoop]
                      forMode:NSRunLoopCommonModes];
}

void iOSWindow::TeardownDisplayLink() {
  if (display_link_) {
    [display_link_ invalidate];
    display_link_ = nil;
  }
  g_display_link_target = nil;
}

}  // namespace ui
}  // namespace xe
