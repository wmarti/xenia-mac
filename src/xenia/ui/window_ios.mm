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

  // Report size in points to match Surface::GetSize semantics.
  // MetalPresenter applies the scale factor when computing drawable size.
  CGRect bounds = [view_ bounds];
  uint32_t width = static_cast<uint32_t>(bounds.size.width);
  uint32_t height = static_cast<uint32_t>(bounds.size.height);

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

  // Report initial size in points (logical coordinates).
  // MetalPresenter applies the backing scale factor for drawable size.
  CGRect bounds = [view_ bounds];
  CGFloat scale = [view_ contentScaleFactor];
  uint32_t width = static_cast<uint32_t>(bounds.size.width);
  uint32_t height = static_cast<uint32_t>(bounds.size.height);
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
  CADisplayLink* _displayLink;
}
- (instancetype)initWithWindow:(xe::ui::iOSWindow*)window;
- (void)displayLinkFired:(CADisplayLink*)link;
- (void)setDisplayLink:(CADisplayLink*)link;
- (void)startObservingLifecycle;
- (void)stopObservingLifecycle;
@end

@implementation XeniaDisplayLinkTarget
- (instancetype)initWithWindow:(xe::ui::iOSWindow*)window {
  if (self = [super init]) {
    _window = window;
    _displayLink = nil;
  }
  return self;
}

- (void)setDisplayLink:(CADisplayLink*)link {
  _displayLink = link;
}

- (void)displayLinkFired:(CADisplayLink*)link {
  if (_window) {
    _window->TriggerPaint();
  }
}

- (void)startObservingLifecycle {
  [[NSNotificationCenter defaultCenter] addObserver:self
                                           selector:@selector(appWillResignActive:)
                                               name:UIApplicationWillResignActiveNotification
                                             object:nil];
  [[NSNotificationCenter defaultCenter] addObserver:self
                                           selector:@selector(appDidBecomeActive:)
                                               name:UIApplicationDidBecomeActiveNotification
                                             object:nil];
}

- (void)stopObservingLifecycle {
  [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)appWillResignActive:(NSNotification*)notification {
  // Pause the display link when the app goes to background to prevent
  // unnecessary GPU/CPU work and avoid iOS termination for background usage.
  if (_displayLink) {
    _displayLink.paused = YES;
  }
  XELOGI("iOS lifecycle: display link paused (app resigned active)");
}

- (void)appDidBecomeActive:(NSNotification*)notification {
  // Resume the display link when the app returns to foreground.
  if (_displayLink) {
    _displayLink.paused = NO;
  }
  // Notify the window of potential size changes (e.g. split-view transitions).
  if (_window) {
    _window->HandleSizeChange();
  }
  XELOGI("iOS lifecycle: display link resumed (app became active)");
}

- (void)dealloc {
  [self stopObservingLifecycle];
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
  [g_display_link_target setDisplayLink:display_link_];
  // Prefer 60 FPS but allow the system to adjust.
  display_link_.preferredFrameRateRange =
      CAFrameRateRangeMake(30.0, 120.0, 60.0);
  [display_link_ addToRunLoop:[NSRunLoop mainRunLoop]
                      forMode:NSRunLoopCommonModes];

  // Observe iOS lifecycle events to pause/resume display link.
  [g_display_link_target startObservingLifecycle];
}

void iOSWindow::TeardownDisplayLink() {
  if (g_display_link_target) {
    [g_display_link_target stopObservingLifecycle];
  }
  if (display_link_) {
    [display_link_ invalidate];
    display_link_ = nil;
  }
  g_display_link_target = nil;
}

}  // namespace ui
}  // namespace xe
