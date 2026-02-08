/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <memory>
#include <string>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/ui/surface_ios.h"
#include "xenia/ui/windowed_app.h"
#include "xenia/ui/windowed_app_context_ios.h"

// Forward declarations of the Objective-C classes.
@class XeniaAppDelegate;
@class XeniaViewController;
@class XeniaMetalView;

// ---------------------------------------------------------------------------
// XeniaMetalView - a UIView backed by a CAMetalLayer.
// ---------------------------------------------------------------------------
@interface XeniaMetalView : UIView
@end

@implementation XeniaMetalView

+ (Class)layerClass {
  return [CAMetalLayer class];
}

@end

// ---------------------------------------------------------------------------
// XeniaViewController - manages the Metal view and game launcher UI.
// ---------------------------------------------------------------------------
@interface XeniaViewController
    : UIViewController <UIDocumentPickerDelegate>
@property(nonatomic, strong) XeniaMetalView* metalView;
@property(nonatomic, strong) UIView* launcherOverlay;
@property(nonatomic, strong) UIButton* openGameButton;
@property(nonatomic, strong) UILabel* titleLabel;
@property(nonatomic, strong) UILabel* statusLabel;
@property(nonatomic, assign) xe::ui::IOSWindowedAppContext* appContext;
@end

@implementation XeniaViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  self.view.backgroundColor = [UIColor blackColor];

  // Create the Metal-backed rendering view (full screen, behind everything).
  self.metalView = [[XeniaMetalView alloc] initWithFrame:self.view.bounds];
  self.metalView.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.metalView.contentScaleFactor = [UIScreen mainScreen].scale;
  [self.view addSubview:self.metalView];

  // Create the launcher overlay UI.
  [self setupLauncherOverlay];
}

- (void)setupLauncherOverlay {
  self.launcherOverlay =
      [[UIView alloc] initWithFrame:self.view.bounds];
  self.launcherOverlay.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.launcherOverlay.backgroundColor =
      [UIColor colorWithWhite:0.0 alpha:0.85];
  [self.view addSubview:self.launcherOverlay];

  // Title label.
  self.titleLabel = [[UILabel alloc] init];
  self.titleLabel.text = @"Xenia";
  self.titleLabel.textColor = [UIColor whiteColor];
  self.titleLabel.font = [UIFont systemFontOfSize:42 weight:UIFontWeightBold];
  self.titleLabel.textAlignment = NSTextAlignmentCenter;
  self.titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:self.titleLabel];

  // Subtitle label.
  UILabel* subtitleLabel = [[UILabel alloc] init];
  subtitleLabel.text = @"Xbox 360 Emulator";
  subtitleLabel.textColor = [UIColor lightGrayColor];
  subtitleLabel.font = [UIFont systemFontOfSize:16 weight:UIFontWeightRegular];
  subtitleLabel.textAlignment = NSTextAlignmentCenter;
  subtitleLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:subtitleLabel];

  // Open Game button.
  UIButtonConfiguration* config =
      [UIButtonConfiguration filledButtonConfiguration];
  config.title = @"Open Game";
  config.image = [UIImage systemImageNamed:@"folder"];
  config.imagePadding = 8;
  config.baseBackgroundColor = [UIColor systemGreenColor];
  config.baseForegroundColor = [UIColor whiteColor];
  config.cornerStyle = UIButtonConfigurationCornerStyleLarge;
  config.contentInsets = NSDirectionalEdgeInsetsMake(14, 32, 14, 32);

  self.openGameButton = [UIButton buttonWithConfiguration:config
                                            primaryAction:nil];
  self.openGameButton.translatesAutoresizingMaskIntoConstraints = NO;
  [self.openGameButton addTarget:self
                          action:@selector(openGameTapped:)
                forControlEvents:UIControlEventTouchUpInside];
  [self.launcherOverlay addSubview:self.openGameButton];

  // Status label (for showing loading state).
  self.statusLabel = [[UILabel alloc] init];
  self.statusLabel.text = @"";
  self.statusLabel.textColor = [UIColor lightGrayColor];
  self.statusLabel.font =
      [UIFont systemFontOfSize:14 weight:UIFontWeightRegular];
  self.statusLabel.textAlignment = NSTextAlignmentCenter;
  self.statusLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:self.statusLabel];

  // Layout constraints.
  [NSLayoutConstraint activateConstraints:@[
    [self.titleLabel.centerXAnchor
        constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.titleLabel.bottomAnchor
        constraintEqualToAnchor:subtitleLabel.topAnchor
                       constant:-4],

    [subtitleLabel.centerXAnchor
        constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [subtitleLabel.bottomAnchor
        constraintEqualToAnchor:self.openGameButton.topAnchor
                       constant:-32],

    [self.openGameButton.centerXAnchor
        constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.openGameButton.centerYAnchor
        constraintEqualToAnchor:self.launcherOverlay.centerYAnchor
                       constant:20],

    [self.statusLabel.centerXAnchor
        constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.statusLabel.topAnchor
        constraintEqualToAnchor:self.openGameButton.bottomAnchor
                       constant:20],
  ]];
}

- (void)openGameTapped:(UIButton*)sender {
  NSArray<UTType*>* contentTypes = @[
    [UTType typeWithFilenameExtension:@"iso"],
    [UTType typeWithFilenameExtension:@"xex"],
    [UTType typeWithFilenameExtension:@"zar"],
    UTTypeData,
  ];

  UIDocumentPickerViewController* picker =
      [[UIDocumentPickerViewController alloc]
          initForOpeningContentTypes:contentTypes];
  picker.delegate = self;
  picker.allowsMultipleSelection = NO;
  picker.shouldShowFileExtensions = YES;
  [self presentViewController:picker animated:YES completion:nil];
}

#pragma mark - UIDocumentPickerDelegate

- (void)documentPicker:(UIDocumentPickerViewController*)controller
    didPickDocumentsAtURLs:(NSArray<NSURL*>*)urls {
  if (urls.count == 0) return;

  NSURL* url = urls[0];
  // Start security-scoped access for files outside the sandbox.
  [url startAccessingSecurityScopedResource];

  NSString* path = url.path;
  XELOGI("iOS: User selected game file: {}", [path UTF8String]);

  // Update UI to show loading state.
  self.statusLabel.text =
      [NSString stringWithFormat:@"Loading: %@", url.lastPathComponent];
  self.openGameButton.enabled = NO;

  // Hide the launcher overlay with animation.
  [UIView animateWithDuration:0.3
      animations:^{
        self.launcherOverlay.alpha = 0.0;
      }
      completion:^(BOOL finished) {
        self.launcherOverlay.hidden = YES;
      }];

  // Launch the game through the app context callback.
  if (self.appContext) {
    std::string game_path = std::string([path UTF8String]);
    self.appContext->LaunchGame(game_path);
  }
}

- (void)documentPickerWasCancelled:
    (UIDocumentPickerViewController*)controller {
  XELOGI("iOS: Document picker cancelled");
}

#pragma mark - Status bar / home indicator

- (BOOL)prefersStatusBarHidden {
  return YES;
}

- (BOOL)prefersHomeIndicatorAutoHidden {
  return YES;
}

- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures {
  return UIRectEdgeAll;
}

#pragma mark - Public API

- (void)showLauncherOverlay {
  self.launcherOverlay.hidden = NO;
  self.openGameButton.enabled = YES;
  self.statusLabel.text = @"";
  [UIView animateWithDuration:0.3
                   animations:^{
                     self.launcherOverlay.alpha = 1.0;
                   }];
}

@end

// ---------------------------------------------------------------------------
// XeniaAppDelegate - UIKit application lifecycle.
// ---------------------------------------------------------------------------
@interface XeniaAppDelegate : UIResponder <UIApplicationDelegate>
@property(nonatomic, strong) UIWindow* window;
@end

@implementation XeniaAppDelegate {
  std::unique_ptr<xe::ui::IOSWindowedAppContext> app_context_;
  std::unique_ptr<xe::ui::WindowedApp> app_;
}

- (BOOL)application:(UIApplication*)application
    didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
  // Initialize cvars with no arguments on iOS (arguments come from config).
  int argc = 1;
  char arg0[] = "xenia_edge";
  char* argv[] = {arg0};
  char** argv_ptr = argv;
  cvar::ParseLaunchArguments(argc, argv_ptr, "", {});

  // Create the app context.
  app_context_ = std::make_unique<xe::ui::IOSWindowedAppContext>();

  // Set up the UIKit window and view controller FIRST, so the Metal view
  // is available when the app initializes.
  self.window = [[UIWindow alloc]
      initWithFrame:[[UIScreen mainScreen] bounds]];
  XeniaViewController* vc = [[XeniaViewController alloc] init];
  self.window.rootViewController = vc;
  [self.window makeKeyAndVisible];

  // Force layout so the Metal view is created.
  [vc.view layoutIfNeeded];

  // Store the Metal view and view controller in the app context for
  // iOSWindow to use.
  app_context_->set_metal_view(vc.metalView);
  app_context_->set_view_controller(vc);
  vc.appContext = app_context_.get();

  XELOGI("iOS: Metal view ready ({}x{})",
         static_cast<uint32_t>(vc.metalView.bounds.size.width *
                               vc.metalView.contentScaleFactor),
         static_cast<uint32_t>(vc.metalView.bounds.size.height *
                               vc.metalView.contentScaleFactor));

  // Create and initialize the Xenia app.
  app_ = xe::ui::GetWindowedAppCreator()(*app_context_);
  xe::InitializeLogging(app_->GetName());

  if (!app_->OnInitialize()) {
    XELOGE("iOS: App initialization failed");
    return NO;
  }

  XELOGI("iOS: Application launched successfully");
  return YES;
}

- (void)applicationWillTerminate:(UIApplication*)application {
  if (app_) {
    app_->InvokeOnDestroy();
    app_.reset();
  }
  app_context_.reset();
}

@end

// ---------------------------------------------------------------------------
// iOS entry point.
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  @autoreleasepool {
    return UIApplicationMain(argc, argv, nil,
                             NSStringFromClass([XeniaAppDelegate class]));
  }
}
