/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/sdl/sdl_audio_system.h"

#include "xenia/apu/apu_flags.h"
#include "xenia/apu/sdl/sdl_audio_driver.h"
#include "xenia/base/logging.h"
#include "xenia/base/threading.h"

namespace xe {
namespace apu {
namespace sdl {

namespace {

#if XE_PLATFORM_IOS
// Fallback used when SDL/CoreAudio device creation fails on iOS.
// Keeps guest audio callback flow alive without real audio output.
class SilentAudioDriver final : public AudioDriver {
 public:
  explicit SilentAudioDriver(xe::threading::Semaphore* semaphore)
      : semaphore_(semaphore) {}

  bool Initialize() override { return true; }
  void Shutdown() override {}
  void SubmitFrame(float* samples) override {
    (void)samples;
    if (semaphore_) {
      semaphore_->Release(1, nullptr);
    }
  }
  void Pause() override {}
  void Resume() override {}
  void SetVolume(float volume) override { (void)volume; }

 private:
  xe::threading::Semaphore* semaphore_ = nullptr;
};
#endif  // XE_PLATFORM_IOS

}  // namespace

std::unique_ptr<AudioSystem> SDLAudioSystem::Create(cpu::Processor* processor) {
  return std::make_unique<SDLAudioSystem>(processor);
}

SDLAudioSystem::SDLAudioSystem(cpu::Processor* processor)
    : AudioSystem(processor) {}

SDLAudioSystem::~SDLAudioSystem() {}

void SDLAudioSystem::Initialize() { AudioSystem::Initialize(); }

X_STATUS SDLAudioSystem::CreateDriver(size_t index,
                                      xe::threading::Semaphore* semaphore,
                                      AudioDriver** out_driver) {
  assert_not_null(out_driver);
  auto driver = std::make_unique<SDLAudioDriver>(semaphore);
  if (!driver->Initialize()) {
    driver->Shutdown();
#if XE_PLATFORM_IOS
    XELOGW("SDLAudioSystem: SDL audio init failed, using silent fallback");
    *out_driver = new SilentAudioDriver(semaphore);
    return X_STATUS_SUCCESS;
#else
    return X_STATUS_UNSUCCESSFUL;
#endif
  }

  *out_driver = driver.release();
  return X_STATUS_SUCCESS;
}

AudioDriver* SDLAudioSystem::CreateDriver(xe::threading::Semaphore* semaphore,
                                          uint32_t frequency, uint32_t channels,
                                          bool need_format_conversion) {
  return new SDLAudioDriver(semaphore, frequency, channels,
                            need_format_conversion);
}

void SDLAudioSystem::DestroyDriver(AudioDriver* driver) {
  assert_not_null(driver);
#if XE_PLATFORM_IOS
  if (auto* silent_driver = dynamic_cast<SilentAudioDriver*>(driver)) {
    silent_driver->Shutdown();
    delete silent_driver;
    return;
  }
#endif
  auto sdldriver = dynamic_cast<SDLAudioDriver*>(driver);
  assert_not_null(sdldriver);
  sdldriver->Shutdown();
  delete sdldriver;
}

}  // namespace sdl
}  // namespace apu
}  // namespace xe
