/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#import <Foundation/Foundation.h>

#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/threading.h"
#include "xenia/config.h"
#include "xenia/emulator.h"
#include "xenia/ui/window.h"
#include "xenia/ui/window_ios.h"
#include "xenia/ui/windowed_app.h"
#include "xenia/ui/windowed_app_context_ios.h"

// Graphics system.
#include "xenia/gpu/metal/metal_graphics_system.h"

// Audio systems.
#include "xenia/apu/nop/nop_audio_system.h"

// Input drivers.
#include "xenia/hid/nop/nop_hid.h"

// CVars normally defined in xenia_main.cc (excluded on iOS).
DEFINE_path(
    storage_root, "",
    "Root path for persistent internal data storage (config, etc.), or empty "
    "to use the path preferred for the OS.",
    "Storage");
DEFINE_path(
    content_root, "",
    "Root path for guest content storage (saves, etc.), or empty to use the "
    "content folder under the storage root.",
    "Storage");
DEFINE_path(
    cache_root, "",
    "Root path for cache files. If empty, the cache folder under the storage "
    "root will be used.",
    "Storage");
DEFINE_bool(mount_scratch, false, "Enable scratch mount", "Storage");
DEFINE_bool(mount_cache, true, "Enable cache mount", "Storage");

// CVar normally defined in windowed_app_main_qt.cc (excluded on iOS).
DEFINE_transient_path(target, "", "Specifies the target file to run.",
                      "General");

namespace xe {
namespace app {

class EmulatorAppIOS final : public xe::ui::WindowedApp {
 public:
  static std::unique_ptr<xe::ui::WindowedApp> Create(
      xe::ui::WindowedAppContext& app_context) {
    return std::unique_ptr<xe::ui::WindowedApp>(
        new EmulatorAppIOS(app_context));
  }

  bool OnInitialize() override;
  void OnDestroy() override;

 private:
  static constexpr size_t kZOrderHidInput = 128;

  explicit EmulatorAppIOS(xe::ui::WindowedAppContext& app_context)
      : xe::ui::WindowedApp(app_context, "xenia") {}

  std::filesystem::path GetIOSStorageRoot() const;
  void EmulatorThread(const std::filesystem::path& game_path);

  static std::unique_ptr<apu::AudioSystem> CreateAudioSystem(
      cpu::Processor* processor);
  static std::unique_ptr<gpu::GraphicsSystem> CreateGraphicsSystem();
  static std::vector<std::unique_ptr<hid::InputDriver>> CreateInputDrivers(
      ui::Window* window);

  std::unique_ptr<Emulator> emulator_;
  std::unique_ptr<ui::Window> window_;
  std::thread emulator_thread_;
  bool emulator_initialized_ = false;
};

std::filesystem::path EmulatorAppIOS::GetIOSStorageRoot() const {
#if XE_PLATFORM_IOS
  // On iOS, use the app's Documents directory as the storage root.
  // This is accessible via Files.app and iTunes file sharing.
  @autoreleasepool {
    NSArray* paths = NSSearchPathForDirectoriesInDomains(
        NSDocumentDirectory, NSUserDomainMask, YES);
    if (paths.count > 0) {
      NSString* documents_path = paths[0];
      return std::filesystem::path([documents_path UTF8String]);
    }
    return std::filesystem::path([NSHomeDirectory() UTF8String]) / "Documents";
  }
#else
  return xe::filesystem::GetUserFolder() / "Xenia";
#endif
}

bool EmulatorAppIOS::OnInitialize() {
  // Set up storage paths.
  std::filesystem::path storage_root = cvars::storage_root;
  if (storage_root.empty()) {
    storage_root = GetIOSStorageRoot();
  }
  storage_root = std::filesystem::absolute(storage_root);

  // Create storage directories if they don't exist.
  std::filesystem::create_directories(storage_root);
  XELOGI("iOS: Storage root: {}", storage_root);

  config::SetupConfig(storage_root);

  std::filesystem::path content_root = cvars::content_root;
  if (content_root.empty()) {
    content_root = storage_root / "content";
  } else if (!content_root.is_absolute()) {
    content_root = storage_root / content_root;
  }
  content_root = std::filesystem::absolute(content_root);
  std::filesystem::create_directories(content_root);
  XELOGI("iOS: Content root: {}", content_root);

  std::filesystem::path cache_root = cvars::cache_root;
  if (cache_root.empty()) {
#if XE_PLATFORM_IOS
    @autoreleasepool {
      NSArray* cache_paths = NSSearchPathForDirectoriesInDomains(
          NSCachesDirectory, NSUserDomainMask, YES);
      if (cache_paths.count > 0) {
        cache_root = std::filesystem::path(
            [cache_paths[0] UTF8String]) / "xenia";
      } else {
        cache_root = storage_root / "cache_host";
      }
    }
#else
    cache_root = storage_root / "cache_host";
#endif
  } else if (!cache_root.is_absolute()) {
    cache_root = storage_root / cache_root;
  }
  cache_root = std::filesystem::absolute(cache_root);
  std::filesystem::create_directories(cache_root);
  XELOGI("iOS: Cache root: {}", cache_root);

  // Create the emulator instance.
  emulator_ =
      std::make_unique<Emulator>("", storage_root, content_root, cache_root);

  // Create the display window from the Metal view in the app context.
  window_ = ui::Window::Create(app_context(), "Xenia", 1280, 720, true);
  if (!window_ || !window_->Open()) {
    XELOGE("iOS: Failed to create or open display window");
    return false;
  }

  XELOGI("iOS: Display window created ({}x{})",
         window_->GetActualPhysicalWidth(),
         window_->GetActualPhysicalHeight());

  // Register callbacks with the app context.
  auto& ios_context =
      static_cast<ui::IOSWindowedAppContext&>(app_context());

  // Forward layout changes (rotation, resize) to the window.
  ios_context.set_layout_changed_callback([this]() {
    if (window_) {
      static_cast<ui::iOSWindow*>(window_.get())->HandleSizeChange();
    }
  });

  ios_context.set_game_launch_callback(
      [this](const std::string& path) {
        XELOGI("iOS: Game launch requested: {}", path);
        auto game_path = std::filesystem::path(path);

        if (emulator_thread_.joinable()) {
          XELOGW("iOS: Emulator thread already running");
          return;
        }

        emulator_thread_ = std::thread(
            [this, game_path]() { EmulatorThread(game_path); });
      });

  XELOGI("iOS: EmulatorAppIOS initialized successfully");
  return true;
}

void EmulatorAppIOS::OnDestroy() {
  if (emulator_thread_.joinable()) {
    if (emulator_) {
      emulator_->TerminateTitle();
    }
    emulator_thread_.join();
  }
  emulator_.reset();
  window_.reset();
}

void EmulatorAppIOS::EmulatorThread(
    const std::filesystem::path& game_path) {
  xe::threading::set_name("Emulator Thread");

  // Load game-specific config if available.
  if (!game_path.empty()) {
    config::LoadGameConfigForFile(game_path);
  }

  // Set up the emulator with all subsystems.
  X_STATUS result = emulator_->Setup(
      window_.get(),
      nullptr,  // No ImGui drawer on iOS for now.
      true,     // require_cpu_backend
      CreateAudioSystem,
      CreateGraphicsSystem,
      CreateInputDrivers);

  if (XFAILED(result)) {
    XELOGE("iOS: Emulator::Setup failed with status {:08X}", result);
    return;
  }

  emulator_initialized_ = true;
  XELOGI("iOS: Emulator setup complete");

  if (!game_path.empty()) {
    auto abs_path = std::filesystem::absolute(game_path);
    XELOGI("iOS: Launching game: {}", abs_path);

    result = emulator_->LaunchPath(abs_path);
    if (XFAILED(result)) {
      XELOGE("iOS: Failed to launch game: {:08X}", result);
      return;
    }

    XELOGI("iOS: Game launched successfully");
    emulator_->WaitUntilExit();
    XELOGI("iOS: Game execution finished");
  }
}

std::unique_ptr<apu::AudioSystem> EmulatorAppIOS::CreateAudioSystem(
    cpu::Processor* processor) {
  // iOS: Use nop audio for now. A CoreAudio / AVAudioEngine backend
  // can be added later for real audio output.
  return std::make_unique<apu::nop::NopAudioSystem>(processor);
}

std::unique_ptr<gpu::GraphicsSystem> EmulatorAppIOS::CreateGraphicsSystem() {
  return std::make_unique<gpu::metal::MetalGraphicsSystem>();
}

std::vector<std::unique_ptr<hid::InputDriver>>
EmulatorAppIOS::CreateInputDrivers(ui::Window* window) {
  std::vector<std::unique_ptr<hid::InputDriver>> drivers;
  // iOS: Use nop input for now. A GCController-based driver can be
  // added later for MFi and Bluetooth gamepad support.
  drivers.emplace_back(
      xe::hid::nop::Create(window, kZOrderHidInput));
  return drivers;
}

}  // namespace app
}  // namespace xe

XE_DEFINE_WINDOWED_APP(xenia, xe::app::EmulatorAppIOS::Create);
