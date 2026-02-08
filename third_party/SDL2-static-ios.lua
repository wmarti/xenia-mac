-- SDL2 static library build for iOS.
-- Only compiles subsystems needed by Xenia: audio (CoreAudio) and
-- gamecontroller/joystick (MFi). Does NOT compile the UIKit video driver
-- to avoid conflicting with Xenia's own UIKit window management.

group("third_party")
project("SDL2")
  uuid("65878768-95A1-4E7B-8DFB-B23A09E0DE7D")
  kind("StaticLib")
  language("C")

  defines({
    "HAVE_LIBC",
    "SDL_LEAN_AND_MEAN",
    "SDL_RENDER_DISABLED",
    "SDL_VIDEO_DRIVER_DUMMY",
    "USING_PREMAKE_CONFIG_H",
    -- Use the iOS-specific SDL config.
    "SDL_config_h_",
  })
  buildoptions({
    "-fobjc-arc",
    "-include", "SDL_config_iphoneos.h",
  })
  includedirs({
    "SDL2/include",
    "SDL2/src",
  })

  -- Core SDL files.
  files({
    "SDL2/src/SDL.c",
    "SDL2/src/SDL_assert.c",
    "SDL2/src/SDL_dataqueue.c",
    "SDL2/src/SDL_error.c",
    "SDL2/src/SDL_guid.c",
    "SDL2/src/SDL_hints.c",
    "SDL2/src/SDL_list.c",
    "SDL2/src/SDL_log.c",
    "SDL2/src/SDL_utils.c",
  })

  -- Atomic.
  files({
    "SDL2/src/atomic/SDL_atomic.c",
    "SDL2/src/atomic/SDL_spinlock.c",
  })

  -- Audio: core + CoreAudio backend.
  files({
    "SDL2/src/audio/SDL_audio.c",
    "SDL2/src/audio/SDL_audiocvt.c",
    "SDL2/src/audio/SDL_audiodev.c",
    "SDL2/src/audio/SDL_audiotypecvt.c",
    "SDL2/src/audio/SDL_mixer.c",
    "SDL2/src/audio/SDL_wave.c",
    "SDL2/src/audio/coreaudio/SDL_coreaudio.m",
    "SDL2/src/audio/dummy/SDL_dummyaudio.c",
  })

  -- CPU info.
  files({
    "SDL2/src/cpuinfo/SDL_cpuinfo.c",
  })

  -- Dynamic API.
  files({
    "SDL2/src/dynapi/SDL_dynapi.c",
  })

  -- Events.
  files({
    "SDL2/src/events/SDL_clipboardevents.c",
    "SDL2/src/events/SDL_displayevents.c",
    "SDL2/src/events/SDL_dropevents.c",
    "SDL2/src/events/SDL_events.c",
    "SDL2/src/events/SDL_gesture.c",
    "SDL2/src/events/SDL_keyboard.c",
    "SDL2/src/events/SDL_mouse.c",
    "SDL2/src/events/SDL_quit.c",
    "SDL2/src/events/SDL_touch.c",
    "SDL2/src/events/SDL_windowevents.c",
  })

  -- File I/O.
  files({
    "SDL2/src/file/SDL_rwops.c",
    "SDL2/src/file/cocoa/SDL_rwopsbundlesupport.m",
  })

  -- Filesystem (cocoa).
  files({
    "SDL2/src/filesystem/cocoa/SDL_sysfilesystem.m",
  })

  -- Haptic (dummy - we only need gamecontroller).
  files({
    "SDL2/src/haptic/SDL_haptic.c",
    "SDL2/src/haptic/dummy/SDL_syshaptic.c",
  })

  -- HID API.
  files({
    "SDL2/src/hidapi/SDL_hidapi.c",
  })

  -- Joystick: core + MFi (iOS GameController framework) backend.
  files({
    "SDL2/src/joystick/SDL_joystick.c",
    "SDL2/src/joystick/SDL_gamecontroller.c",
    "SDL2/src/joystick/SDL_steam_virtual_gamepad.c",
    "SDL2/src/joystick/controller_type.c",
    "SDL2/src/joystick/virtual/SDL_virtualjoystick.c",
    "SDL2/src/joystick/iphoneos/SDL_mfijoystick.m",
    "SDL2/src/joystick/hidapi/SDL_hidapijoystick.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_combined.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_gamecube.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_luna.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_ps3.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_ps4.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_ps5.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_rumble.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_shield.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_stadia.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_steam.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_steamdeck.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_switch.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_wii.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_xbox360.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_xbox360w.c",
    "SDL2/src/joystick/hidapi/SDL_hidapi_xboxone.c",
  })

  -- Math library.
  files({
    "SDL2/src/libm/e_atan2.c",
    "SDL2/src/libm/e_exp.c",
    "SDL2/src/libm/e_fmod.c",
    "SDL2/src/libm/e_log.c",
    "SDL2/src/libm/e_log10.c",
    "SDL2/src/libm/e_pow.c",
    "SDL2/src/libm/e_rem_pio2.c",
    "SDL2/src/libm/e_sqrt.c",
    "SDL2/src/libm/k_cos.c",
    "SDL2/src/libm/k_rem_pio2.c",
    "SDL2/src/libm/k_sin.c",
    "SDL2/src/libm/k_tan.c",
    "SDL2/src/libm/s_atan.c",
    "SDL2/src/libm/s_copysign.c",
    "SDL2/src/libm/s_cos.c",
    "SDL2/src/libm/s_fabs.c",
    "SDL2/src/libm/s_floor.c",
    "SDL2/src/libm/s_scalbn.c",
    "SDL2/src/libm/s_sin.c",
    "SDL2/src/libm/s_tan.c",
  })

  -- Dynamic library loading.
  files({
    "SDL2/src/loadso/dlopen/SDL_sysloadso.c",
  })

  -- Locale.
  files({
    "SDL2/src/locale/SDL_locale.c",
    "SDL2/src/locale/macosx/SDL_syslocale.m",
  })

  -- Misc.
  files({
    "SDL2/src/misc/SDL_url.c",
    "SDL2/src/misc/ios/SDL_sysurl.m",
  })

  -- Power.
  files({
    "SDL2/src/power/SDL_power.c",
    "SDL2/src/power/uikit/SDL_syspower.m",
  })

  -- Sensor (dummy).
  files({
    "SDL2/src/sensor/SDL_sensor.c",
    "SDL2/src/sensor/dummy/SDL_dummysensor.c",
  })

  -- Standard library.
  files({
    "SDL2/src/stdlib/SDL_crc16.c",
    "SDL2/src/stdlib/SDL_crc32.c",
    "SDL2/src/stdlib/SDL_getenv.c",
    "SDL2/src/stdlib/SDL_iconv.c",
    "SDL2/src/stdlib/SDL_malloc.c",
    "SDL2/src/stdlib/SDL_qsort.c",
    "SDL2/src/stdlib/SDL_stdlib.c",
    "SDL2/src/stdlib/SDL_string.c",
    "SDL2/src/stdlib/SDL_strtokr.c",
  })

  -- Thread (pthreads).
  files({
    "SDL2/src/thread/SDL_thread.c",
    "SDL2/src/thread/pthread/SDL_syscond.c",
    "SDL2/src/thread/pthread/SDL_sysmutex.c",
    "SDL2/src/thread/pthread/SDL_syssem.c",
    "SDL2/src/thread/pthread/SDL_systhread.c",
    "SDL2/src/thread/pthread/SDL_systls.c",
  })

  -- Timer (unix).
  files({
    "SDL2/src/timer/SDL_timer.c",
    "SDL2/src/timer/unix/SDL_systimer.c",
  })

  -- Video (dummy only - Xenia has its own UIKit/Metal window management).
  files({
    "SDL2/src/video/SDL_blit.c",
    "SDL2/src/video/SDL_blit_0.c",
    "SDL2/src/video/SDL_blit_1.c",
    "SDL2/src/video/SDL_blit_A.c",
    "SDL2/src/video/SDL_blit_N.c",
    "SDL2/src/video/SDL_blit_auto.c",
    "SDL2/src/video/SDL_blit_copy.c",
    "SDL2/src/video/SDL_blit_slow.c",
    "SDL2/src/video/SDL_bmp.c",
    "SDL2/src/video/SDL_clipboard.c",
    "SDL2/src/video/SDL_fillrect.c",
    "SDL2/src/video/SDL_pixels.c",
    "SDL2/src/video/SDL_rect.c",
    "SDL2/src/video/SDL_RLEaccel.c",
    "SDL2/src/video/SDL_shape.c",
    "SDL2/src/video/SDL_stretch.c",
    "SDL2/src/video/SDL_surface.c",
    "SDL2/src/video/SDL_video.c",
    "SDL2/src/video/SDL_yuv.c",
    "SDL2/src/video/dummy/SDL_nullevents.c",
    "SDL2/src/video/dummy/SDL_nullframebuffer.c",
    "SDL2/src/video/dummy/SDL_nullvideo.c",
  })

  -- Core (unix).
  files({
    "SDL2/src/core/unix/SDL_poll.c",
  })
