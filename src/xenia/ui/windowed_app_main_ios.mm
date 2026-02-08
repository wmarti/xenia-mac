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

#include <sys/mman.h>
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
// JIT availability check -- tests whether executable memory can be mapped.
// Requires CS_DEBUGGED (set by StikDebug, AltJIT, SideJITServer, etc.).
// ---------------------------------------------------------------------------
static BOOL xe_check_jit_available(void) {
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) page_size = 16384;
  void* test =
      mmap(NULL, (size_t)page_size, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (test == MAP_FAILED) {
    return NO;
  }
  munmap(test, (size_t)page_size);
  return YES;
}

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
@property(nonatomic, strong) NSURL* securityScopedURL;
@property(nonatomic, assign) xe::ui::IOSWindowedAppContext* appContext;

// JIT gate overlay.
@property(nonatomic, strong) UIView* jitOverlay;
@property(nonatomic, strong) UIView* jitPulseView;
@property(nonatomic, strong) UIView* jitStatusDot;
@property(nonatomic, strong) UILabel* jitStatusLabel;
@property(nonatomic, strong) NSTimer* jitPollTimer;
@property(nonatomic, assign) BOOL jitAcquired;
@end

@implementation XeniaViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  self.view.backgroundColor = [UIColor blackColor];
  self.jitAcquired = NO;

  // Create the Metal-backed rendering view (full screen, behind everything).
  self.metalView = [[XeniaMetalView alloc] initWithFrame:self.view.bounds];
  self.metalView.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.metalView.contentScaleFactor = [UIScreen mainScreen].scale;
  [self.view addSubview:self.metalView];

  // Create the launcher overlay UI (starts hidden until JIT is acquired).
  [self setupLauncherOverlay];
  self.launcherOverlay.hidden = YES;
  self.launcherOverlay.alpha = 0.0;

  // Create the JIT gate overlay (blocks interaction until JIT is available).
  [self setupJITOverlay];

  // Start polling for JIT.
  [self startJITPoll];
}

// ---------------------------------------------------------------------------
// JIT gate overlay -- shown until JIT is acquired.
// ---------------------------------------------------------------------------
- (void)setupJITOverlay {
  self.jitOverlay = [[UIView alloc] initWithFrame:self.view.bounds];
  self.jitOverlay.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.jitOverlay.backgroundColor = [UIColor colorWithWhite:0.0 alpha:0.95];
  [self.view addSubview:self.jitOverlay];

  // Container for centered content.
  UIView* container = [[UIView alloc] init];
  container.translatesAutoresizingMaskIntoConstraints = NO;
  [self.jitOverlay addSubview:container];

  // Pulsing circle behind the icon.
  self.jitPulseView = [[UIView alloc] init];
  self.jitPulseView.translatesAutoresizingMaskIntoConstraints = NO;
  self.jitPulseView.backgroundColor = [UIColor colorWithRed:0.0 green:0.478 blue:1.0 alpha:0.15];
  self.jitPulseView.layer.cornerRadius = 50;
  [container addSubview:self.jitPulseView];

  // CPU icon.
  UIImageView* iconView = [[UIImageView alloc]
      initWithImage:
          [UIImage systemImageNamed:@"cpu"
                  withConfiguration:[UIImageSymbolConfiguration
                                        configurationWithPointSize:50
                                                            weight:UIImageSymbolWeightMedium]]];
  iconView.tintColor = [UIColor systemBlueColor];
  iconView.translatesAutoresizingMaskIntoConstraints = NO;
  iconView.contentMode = UIViewContentModeCenter;
  [container addSubview:iconView];

  // "Waiting for JIT" title.
  UILabel* jitTitle = [[UILabel alloc] init];
  jitTitle.text = @"Waiting for JIT";
  jitTitle.textColor = [UIColor whiteColor];
  jitTitle.font = [UIFont systemFontOfSize:24 weight:UIFontWeightSemibold];
  jitTitle.textAlignment = NSTextAlignmentCenter;
  jitTitle.translatesAutoresizingMaskIntoConstraints = NO;
  [container addSubview:jitTitle];

  // Subtitle.
  UILabel* jitSubtitle = [[UILabel alloc] init];
  jitSubtitle.text = @"Waiting for Just-In-Time compilation...";
  jitSubtitle.textColor = [UIColor secondaryLabelColor];
  jitSubtitle.font = [UIFont systemFontOfSize:15 weight:UIFontWeightRegular];
  jitSubtitle.textAlignment = NSTextAlignmentCenter;
  jitSubtitle.translatesAutoresizingMaskIntoConstraints = NO;
  [container addSubview:jitSubtitle];

  // Info card background.
  UIView* infoCard = [[UIView alloc] init];
  infoCard.translatesAutoresizingMaskIntoConstraints = NO;
  infoCard.backgroundColor = [UIColor colorWithWhite:0.15 alpha:1.0];
  infoCard.layer.cornerRadius = 12;
  [container addSubview:infoCard];

  // Info row 1: blue info icon + description.
  UIImageView* infoIcon =
      [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"info.circle.fill"]];
  infoIcon.tintColor = [UIColor systemBlueColor];
  infoIcon.translatesAutoresizingMaskIntoConstraints = NO;
  [infoCard addSubview:infoIcon];

  UILabel* infoText = [[UILabel alloc] init];
  infoText.text = @"JIT compilation is required for Xenia to run "
                  @"Xbox 360 games. It dynamically translates and "
                  @"executes code at full speed.";
  infoText.textColor = [UIColor secondaryLabelColor];
  infoText.font = [UIFont systemFontOfSize:13];
  infoText.numberOfLines = 0;
  infoText.translatesAutoresizingMaskIntoConstraints = NO;
  [infoCard addSubview:infoText];

  // Info row 2: green check icon + instructions.
  UIImageView* checkIcon =
      [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"checkmark.circle.fill"]];
  checkIcon.tintColor = [UIColor systemGreenColor];
  checkIcon.translatesAutoresizingMaskIntoConstraints = NO;
  [infoCard addSubview:checkIcon];

  UILabel* checkText = [[UILabel alloc] init];
  checkText.text = @"Enable JIT using StikDebug, SideJITServer, "
                   @"or AltJIT. If running from Xcode, attach the "
                   @"debugger.";
  checkText.textColor = [UIColor secondaryLabelColor];
  checkText.font = [UIFont systemFontOfSize:13];
  checkText.numberOfLines = 0;
  checkText.translatesAutoresizingMaskIntoConstraints = NO;
  [infoCard addSubview:checkText];

  // Layout.
  CGFloat maxCardWidth = 360;
  [NSLayoutConstraint activateConstraints:@[
    // Container centered in overlay.
    [container.centerXAnchor constraintEqualToAnchor:self.jitOverlay.centerXAnchor],
    [container.centerYAnchor constraintEqualToAnchor:self.jitOverlay.centerYAnchor],
    [container.widthAnchor constraintLessThanOrEqualToConstant:maxCardWidth],
    [container.leadingAnchor constraintGreaterThanOrEqualToAnchor:self.jitOverlay.leadingAnchor
                                                         constant:24],

    // Pulse circle.
    [self.jitPulseView.centerXAnchor constraintEqualToAnchor:container.centerXAnchor],
    [self.jitPulseView.topAnchor constraintEqualToAnchor:container.topAnchor],
    [self.jitPulseView.widthAnchor constraintEqualToConstant:100],
    [self.jitPulseView.heightAnchor constraintEqualToConstant:100],

    // Icon centered on pulse.
    [iconView.centerXAnchor constraintEqualToAnchor:self.jitPulseView.centerXAnchor],
    [iconView.centerYAnchor constraintEqualToAnchor:self.jitPulseView.centerYAnchor],

    // Title below icon.
    [jitTitle.topAnchor constraintEqualToAnchor:self.jitPulseView.bottomAnchor constant:20],
    [jitTitle.centerXAnchor constraintEqualToAnchor:container.centerXAnchor],

    // Subtitle below title.
    [jitSubtitle.topAnchor constraintEqualToAnchor:jitTitle.bottomAnchor constant:8],
    [jitSubtitle.centerXAnchor constraintEqualToAnchor:container.centerXAnchor],

    // Info card below subtitle.
    [infoCard.topAnchor constraintEqualToAnchor:jitSubtitle.bottomAnchor constant:24],
    [infoCard.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
    [infoCard.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
    [infoCard.bottomAnchor constraintEqualToAnchor:container.bottomAnchor],

    // Info row 1.
    [infoIcon.leadingAnchor constraintEqualToAnchor:infoCard.leadingAnchor constant:14],
    [infoIcon.topAnchor constraintEqualToAnchor:infoCard.topAnchor constant:14],
    [infoIcon.widthAnchor constraintEqualToConstant:18],
    [infoIcon.heightAnchor constraintEqualToConstant:18],

    [infoText.leadingAnchor constraintEqualToAnchor:infoIcon.trailingAnchor constant:10],
    [infoText.trailingAnchor constraintEqualToAnchor:infoCard.trailingAnchor constant:-14],
    [infoText.topAnchor constraintEqualToAnchor:infoIcon.topAnchor constant:-2],

    // Info row 2.
    [checkIcon.leadingAnchor constraintEqualToAnchor:infoCard.leadingAnchor constant:14],
    [checkIcon.topAnchor constraintEqualToAnchor:infoText.bottomAnchor constant:14],
    [checkIcon.widthAnchor constraintEqualToConstant:18],
    [checkIcon.heightAnchor constraintEqualToConstant:18],

    [checkText.leadingAnchor constraintEqualToAnchor:checkIcon.trailingAnchor constant:10],
    [checkText.trailingAnchor constraintEqualToAnchor:infoCard.trailingAnchor constant:-14],
    [checkText.topAnchor constraintEqualToAnchor:checkIcon.topAnchor constant:-2],
    [checkText.bottomAnchor constraintEqualToAnchor:infoCard.bottomAnchor constant:-14],
  ]];

  // Start the pulsing animation.
  [self startPulseAnimation];
}

- (void)startPulseAnimation {
  self.jitPulseView.alpha = 1.0;
  self.jitPulseView.transform = CGAffineTransformIdentity;

  [UIView animateWithDuration:1.5
                        delay:0.0
                      options:UIViewAnimationOptionCurveEaseInOut | UIViewAnimationOptionRepeat |
                              UIViewAnimationOptionAllowUserInteraction
                   animations:^{
                     self.jitPulseView.transform = CGAffineTransformMakeScale(1.3, 1.3);
                     self.jitPulseView.alpha = 0.0;
                   }
                   completion:nil];
}

// ---------------------------------------------------------------------------
// JIT polling -- checks every 0.5s until JIT is available.
// ---------------------------------------------------------------------------
- (void)startJITPoll {
  // Check immediately first.
  if (xe_check_jit_available()) {
    [self onJITAcquired];
    return;
  }

  XELOGI("iOS: JIT not yet available, polling...");
  self.jitPollTimer = [NSTimer scheduledTimerWithTimeInterval:0.5
                                                       target:self
                                                     selector:@selector(pollJIT:)
                                                     userInfo:nil
                                                      repeats:YES];
}

- (void)pollJIT:(NSTimer*)timer {
  if (xe_check_jit_available()) {
    [timer invalidate];
    self.jitPollTimer = nil;
    [self onJITAcquired];
  }
}

- (void)onJITAcquired {
  self.jitAcquired = YES;
  XELOGI("iOS: JIT acquired!");

  // Update the JIT status indicator on the launcher.
  [self updateJITStatusIndicator];

  // Fade out JIT overlay, reveal launcher.
  self.launcherOverlay.hidden = NO;
  [UIView animateWithDuration:0.4
      delay:0.0
      options:UIViewAnimationOptionCurveEaseOut
      animations:^{
        self.jitOverlay.alpha = 0.0;
        self.launcherOverlay.alpha = 1.0;
      }
      completion:^(BOOL finished) {
        [self.jitOverlay removeFromSuperview];
        self.jitOverlay = nil;
      }];
}

// ---------------------------------------------------------------------------
// Launcher overlay with Open Game button.
// ---------------------------------------------------------------------------
- (void)setupLauncherOverlay {
  self.launcherOverlay =
      [[UIView alloc] initWithFrame:self.view.bounds];
  self.launcherOverlay.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  self.launcherOverlay.backgroundColor =
      [UIColor colorWithWhite:0.0 alpha:0.85];
  [self.view addSubview:self.launcherOverlay];

  // JIT status indicator (green/red dot + label) at the top.
  UIView* statusContainer = [[UIView alloc] init];
  statusContainer.translatesAutoresizingMaskIntoConstraints = NO;
  [self.launcherOverlay addSubview:statusContainer];

  self.jitStatusDot = [[UIView alloc] init];
  self.jitStatusDot.translatesAutoresizingMaskIntoConstraints = NO;
  self.jitStatusDot.backgroundColor = [UIColor systemRedColor];
  self.jitStatusDot.layer.cornerRadius = 5;
  [statusContainer addSubview:self.jitStatusDot];

  self.jitStatusLabel = [[UILabel alloc] init];
  self.jitStatusLabel.text = @"JIT Not Acquired";
  self.jitStatusLabel.textColor = [UIColor systemRedColor];
  self.jitStatusLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightMedium];
  self.jitStatusLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [statusContainer addSubview:self.jitStatusLabel];

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
    // JIT status indicator at top center.
    [statusContainer.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [statusContainer.topAnchor
        constraintEqualToAnchor:self.launcherOverlay.safeAreaLayoutGuide.topAnchor
                       constant:16],

    [self.jitStatusDot.leadingAnchor constraintEqualToAnchor:statusContainer.leadingAnchor],
    [self.jitStatusDot.centerYAnchor constraintEqualToAnchor:statusContainer.centerYAnchor],
    [self.jitStatusDot.widthAnchor constraintEqualToConstant:10],
    [self.jitStatusDot.heightAnchor constraintEqualToConstant:10],

    [self.jitStatusLabel.leadingAnchor constraintEqualToAnchor:self.jitStatusDot.trailingAnchor
                                                      constant:6],
    [self.jitStatusLabel.trailingAnchor constraintEqualToAnchor:statusContainer.trailingAnchor],
    [self.jitStatusLabel.centerYAnchor constraintEqualToAnchor:statusContainer.centerYAnchor],

    // Title and subtitle.
    [self.titleLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.titleLabel.bottomAnchor constraintEqualToAnchor:subtitleLabel.topAnchor constant:-4],

    [subtitleLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [subtitleLabel.bottomAnchor constraintEqualToAnchor:self.openGameButton.topAnchor constant:-32],

    [self.openGameButton.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.openGameButton.centerYAnchor constraintEqualToAnchor:self.launcherOverlay.centerYAnchor
                                                      constant:20],

    [self.statusLabel.centerXAnchor constraintEqualToAnchor:self.launcherOverlay.centerXAnchor],
    [self.statusLabel.topAnchor constraintEqualToAnchor:self.openGameButton.bottomAnchor
                                               constant:20],
  ]];
}

- (void)updateJITStatusIndicator {
  if (self.jitAcquired) {
    self.jitStatusDot.backgroundColor = [UIColor systemGreenColor];
    self.jitStatusLabel.text = @"JIT Enabled";
    self.jitStatusLabel.textColor = [UIColor systemGreenColor];
  } else {
    self.jitStatusDot.backgroundColor = [UIColor systemRedColor];
    self.jitStatusLabel.text = @"JIT Not Acquired";
    self.jitStatusLabel.textColor = [UIColor systemRedColor];
  }
}

- (void)viewDidLayoutSubviews {
  [super viewDidLayoutSubviews];
  // Notify the app context that the layout changed, so the window and
  // presenter can update for rotation, split-view, or safe-area changes.
  if (self.appContext) {
    self.appContext->NotifyLayoutChanged();
  }
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

  // Stop any previous security-scoped access before starting a new one.
  if (self.securityScopedURL) {
    [self.securityScopedURL stopAccessingSecurityScopedResource];
    self.securityScopedURL = nil;
  }

  // Start security-scoped access for files outside the sandbox.
  BOOL accessGranted = [url startAccessingSecurityScopedResource];
  if (accessGranted) {
    self.securityScopedURL = url;
  }

  NSString* path = url.path;
  XELOGI("iOS: User selected game file: {} (security-scoped: {})", [path UTF8String],
         accessGranted ? "yes" : "no");

  // Save a bookmark for potential future relaunch.
  NSError* bookmarkError = nil;
  NSData* bookmark = [url bookmarkDataWithOptions:0
                   includingResourceValuesForKeys:nil
                                    relativeToURL:nil
                                            error:&bookmarkError];
  if (bookmark) {
    [[NSUserDefaults standardUserDefaults] setObject:bookmark forKey:@"lastGameBookmark"];
  }

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
  [self updateJITStatusIndicator];
  [UIView animateWithDuration:0.3
                   animations:^{
                     self.launcherOverlay.alpha = 1.0;
                   }];
}

- (void)dealloc {
  [self.jitPollTimer invalidate];
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
  // Release security-scoped file access.
  XeniaViewController* vc = (XeniaViewController*)self.window.rootViewController;
  if (vc.securityScopedURL) {
    [vc.securityScopedURL stopAccessingSecurityScopedResource];
    vc.securityScopedURL = nil;
  }

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
