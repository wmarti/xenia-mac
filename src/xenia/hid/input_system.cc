/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/logging.h"

#include "xenia/hid/input_system.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "xenia/base/profiling.h"
#include "xenia/hid/hid_flags.h"
#include "xenia/hid/input.h"
#include "xenia/hid/input_driver.h"
#include "xenia/kernel/util/shim_utils.h"

#include "xenia/hid/skylander/skylander_emulated.h"
#ifdef XE_PLATFORM_WIN32
#include "xenia/hid/skylander/skylander_hardware.h"
#endif  // XE_PLATFORM_WIN32

namespace xe {
namespace hid {

DEFINE_bool(vibration, true, "Toggle controller vibration.", "HID");

DEFINE_string(
    auto_input_script, "",
    "Headless debugging: scripted virtual controller on slot 0. Semicolon-"
    "separated list of `seconds:button` events; each button is auto-pulsed for "
    "~250ms at that time. Buttons: a,b,x,y,start,back,guide,up,down,left,right,"
    "lb,rb. Example: \"16:start;19:a;23:down;25:a\".",
    "HID");

DEFINE_string(
    auto_input_file, "",
    "Headless debugging: live virtual controller on slot 0. Path to a file "
    "whose contents are the buttons currently held (space-separated names, or a "
    "`lx/ly/rx/ry=<-1..1>` token for sticks; empty = nothing). Re-read ~every "
    "30ms, so writing a name then clearing it produces a button press. Buttons: "
    "a,b,x,y,start,back,guide,up,down,left,right,lb,rb.",
    "HID");

DEFINE_string(
    record_input, "",
    "Debugging: record slot-0 controller input to this file as a timeline of "
    "`<elapsed_seconds> <tokens>` lines (written whenever the held state "
    "changes). Replay later with --replay_input to reproduce a session.",
    "HID");

DEFINE_string(
    replay_input, "",
    "Debugging: replay a controller timeline previously captured with "
    "--record_input onto virtual slot 0.",
    "HID");

namespace {
// Maps a button name to its X_INPUT_GAMEPAD_BUTTON mask. Returns 0 if unknown.
uint16_t AutoInputButtonMask(const std::string& name) {
  if (name == "a") return X_INPUT_GAMEPAD_A;
  if (name == "b") return X_INPUT_GAMEPAD_B;
  if (name == "x") return X_INPUT_GAMEPAD_X;
  if (name == "y") return X_INPUT_GAMEPAD_Y;
  if (name == "start") return X_INPUT_GAMEPAD_START;
  if (name == "back") return X_INPUT_GAMEPAD_BACK;
  if (name == "guide") return X_INPUT_GAMEPAD_GUIDE;
  if (name == "up") return X_INPUT_GAMEPAD_DPAD_UP;
  if (name == "down") return X_INPUT_GAMEPAD_DPAD_DOWN;
  if (name == "left") return X_INPUT_GAMEPAD_DPAD_LEFT;
  if (name == "right") return X_INPUT_GAMEPAD_DPAD_RIGHT;
  if (name == "lb") return X_INPUT_GAMEPAD_LEFT_SHOULDER;
  if (name == "rb") return X_INPUT_GAMEPAD_RIGHT_SHOULDER;
  return 0;
}

// Builds the button state for the scripted controller at a given elapsed time.
// Each scripted event presses its button for a 250ms window so the game sees a
// clean down/up edge.
uint16_t AutoInputButtonsAt(double elapsed_seconds) {
  constexpr double kPulse = 0.25;
  uint16_t buttons = 0;
  std::stringstream ss(cvars::auto_input_script);
  std::string item;
  while (std::getline(ss, item, ';')) {
    auto colon = item.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    double t = 0.0;
    try {
      t = std::stod(item.substr(0, colon));
    } catch (...) {
      continue;
    }
    if (elapsed_seconds >= t && elapsed_seconds < t + kPulse) {
      buttons |= AutoInputButtonMask(item.substr(colon + 1));
    }
  }
  return buttons;
}

// Shared monotonic clock so record / replay / script timestamps all agree.
double AutoInputElapsed() {
  static const auto start = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

// Parses a whitespace-separated token string into a gamepad. Tokens are button
// names (a,b,...,start) or axis tokens lx/ly/rx/ry/lt/rt=<value>.
void ParseTokensIntoGamepad(const std::string& tokens, X_INPUT_GAMEPAD* gp) {
  std::stringstream ss(tokens);
  std::string token;
  while (ss >> token) {
    auto eq = token.find('=');
    if (eq != std::string::npos) {
      std::string axis = token.substr(0, eq);
      double v = 0.0;
      try {
        v = std::stod(token.substr(eq + 1));
      } catch (...) {
        continue;
      }
      int16_t scaled = int16_t(std::max(-1.0, std::min(1.0, v)) * 32767.0);
      if (axis == "lx") gp->thumb_lx = scaled;
      else if (axis == "ly") gp->thumb_ly = scaled;
      else if (axis == "rx") gp->thumb_rx = scaled;
      else if (axis == "ry") gp->thumb_ry = scaled;
      else if (axis == "lt") gp->left_trigger = uint8_t(std::max(0.0, v) * 255);
      else if (axis == "rt") gp->right_trigger = uint8_t(std::max(0.0, v) * 255);
    } else {
      gp->buttons = uint16_t(gp->buttons) | AutoInputButtonMask(token);
    }
  }
}

// Serializes the meaningful (non-neutral) part of a gamepad to a token string.
std::string SerializeGamepad(const X_INPUT_GAMEPAD& gp) {
  static const std::pair<uint16_t, const char*> kButtons[] = {
      {X_INPUT_GAMEPAD_A, "a"},
      {X_INPUT_GAMEPAD_B, "b"},
      {X_INPUT_GAMEPAD_X, "x"},
      {X_INPUT_GAMEPAD_Y, "y"},
      {X_INPUT_GAMEPAD_START, "start"},
      {X_INPUT_GAMEPAD_BACK, "back"},
      {X_INPUT_GAMEPAD_GUIDE, "guide"},
      {X_INPUT_GAMEPAD_DPAD_UP, "up"},
      {X_INPUT_GAMEPAD_DPAD_DOWN, "down"},
      {X_INPUT_GAMEPAD_DPAD_LEFT, "left"},
      {X_INPUT_GAMEPAD_DPAD_RIGHT, "right"},
      {X_INPUT_GAMEPAD_LEFT_SHOULDER, "lb"},
      {X_INPUT_GAMEPAD_RIGHT_SHOULDER, "rb"},
  };
  uint16_t buttons = gp.buttons;
  std::string out;
  for (const auto& b : kButtons) {
    if (buttons & b.first) {
      if (!out.empty()) out += ' ';
      out += b.second;
    }
  }
  // Record sticks/triggers only when meaningfully deflected, to keep the log
  // readable and edge-triggered on the digital buttons.
  auto add_axis = [&](const char* name, double v) {
    if (std::abs(v) > 0.2) {
      if (!out.empty()) out += ' ';
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%s=%.2f", name, v);
      out += buf;
    }
  };
  add_axis("lx", int16_t(gp.thumb_lx) / 32767.0);
  add_axis("ly", int16_t(gp.thumb_ly) / 32767.0);
  add_axis("rx", int16_t(gp.thumb_rx) / 32767.0);
  add_axis("ry", int16_t(gp.thumb_ry) / 32767.0);
  if (gp.left_trigger > 50) {
    if (!out.empty()) out += ' ';
    out += "lt=1";
  }
  if (gp.right_trigger > 50) {
    if (!out.empty()) out += ' ';
    out += "rt=1";
  }
  return out;
}

// Appends a timeline line to cvars::record_input whenever the held state changes.
void RecordInput(const X_INPUT_GAMEPAD& gp) {
  static std::ofstream out(cvars::record_input, std::ios::trunc);
  static std::string last = "\x01";  // force first write
  std::string cur = SerializeGamepad(gp);
  if (cur != last) {
    last = cur;
    out << AutoInputElapsed() << ' ' << cur << '\n';
    out.flush();
  }
}

// Live-file virtual controller: re-reads cvars::auto_input_file ~every 30ms.
void ApplyLiveInputFile(X_INPUT_GAMEPAD* gp) {
  static std::string cached_contents;
  static std::chrono::steady_clock::time_point last_read{};
  auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration<double>(now - last_read).count() > 0.03) {
    last_read = now;
    std::ifstream f(cvars::auto_input_file);
    if (f) {
      std::stringstream buf;
      buf << f.rdbuf();
      cached_contents = buf.str();
    } else {
      cached_contents.clear();
    }
  }
  ParseTokensIntoGamepad(cached_contents, gp);
}

// Replays a recorded timeline: applies the most recent entry at <= elapsed.
void ApplyReplayInput(double elapsed, X_INPUT_GAMEPAD* gp) {
  static std::vector<std::pair<double, std::string>> timeline = [] {
    std::vector<std::pair<double, std::string>> tl;
    std::ifstream f(cvars::replay_input);
    std::string line;
    while (std::getline(f, line)) {
      auto sp = line.find(' ');
      double t = 0.0;
      try {
        t = std::stod(sp == std::string::npos ? line : line.substr(0, sp));
      } catch (...) {
        continue;
      }
      tl.emplace_back(t, sp == std::string::npos ? "" : line.substr(sp + 1));
    }
    return tl;
  }();
  const std::string* active = nullptr;
  for (const auto& e : timeline) {
    if (e.first <= elapsed) {
      active = &e.second;
    } else {
      break;
    }
  }
  if (active) {
    ParseTokensIntoGamepad(*active, gp);
  }
}

// Synthesizes a virtual slot-0 controller from auto_input_script (timed),
// auto_input_file (live), or replay_input (recorded). Returns true (and fills
// out_state) when a virtual controller is active.
bool MaybeApplyAutoInput(uint32_t user_index, X_INPUT_STATE* out_state) {
  bool have_script = !cvars::auto_input_script.empty();
  bool have_file = !cvars::auto_input_file.empty();
  bool have_replay = !cvars::replay_input.empty();
  if ((!have_script && !have_file && !have_replay) || user_index != 0 ||
      !out_state) {
    return false;
  }
  static uint32_t packet = 0;
  double elapsed = AutoInputElapsed();
  std::memset(out_state, 0, sizeof(*out_state));
  out_state->packet_number = ++packet;
  if (have_script) {
    out_state->gamepad.buttons = AutoInputButtonsAt(elapsed);
  }
  if (have_file) {
    ApplyLiveInputFile(&out_state->gamepad);
  }
  if (have_replay) {
    ApplyReplayInput(elapsed, &out_state->gamepad);
  }
  return true;
}
}  // namespace

DEFINE_double(left_stick_deadzone_percentage, 0.0,
              "Defines deadzone level for left stick. Allowed range [0.0-1.0].",
              "HID");
DEFINE_double(
    right_stick_deadzone_percentage, 0.0,
    "Defines deadzone level for right stick. Allowed range [0.0-1.0].", "HID");

InputSystem::InputSystem(xe::ui::Window* window) : window_(window) {
  skylander_portal_ = std::make_unique<SkylanderPortalEmulated>();

#ifdef XE_PLATFORM_WIN32
  skylander_portal_ = std::make_unique<SkylanderPortalLibusb>();
#endif  // XE_PLATFORM_WIN32
}

InputSystem::~InputSystem() = default;

X_STATUS InputSystem::Setup() { return X_STATUS_SUCCESS; }

void InputSystem::AddDriver(std::unique_ptr<InputDriver> driver) {
  drivers_.push_back(std::move(driver));
}

void InputSystem::UpdateUsedSlot(InputDriver* driver, uint8_t slot,
                                 bool connected) {
  if (slot == XUserIndexAny) {
    XELOGW("{} received requrest for slot any! Unsupported", __func__);
    return;
  }

  // Do not report passthrough as a controller.
  if (driver && driver->GetInputType() == InputType::Keyboard) {
    return;
  }

  if (connected_slots.test(slot) == connected) {
    // No state change, so nothing to do.
    return;
  }

  XELOGI(controller_slot_state_change_message[connected].c_str(), slot);
  connected_slots.flip(slot);
  // NOTE: previously suppressed on macOS. Games (e.g. Gears of War) gate menu
  // input on this notification, so without it a real or virtual controller is
  // seen as "Controller Not Connected" and input is ignored. The kernel_state
  // null check below guards the not-yet-ready case.
  if (kernel::kernel_state()) {
    kernel::kernel_state()->BroadcastNotification(
        kXNotificationSystemInputDevicesChanged, 0);
  }

  if (driver) {
    X_INPUT_CAPABILITIES capabilities = {};
    const X_RESULT result = driver->GetCapabilities(slot, 0, &capabilities);
    if (result != X_STATUS_SUCCESS) {
      return;
    }

    controllers_max_joystick_value[slot] = {
        {capabilities.gamepad.thumb_lx, capabilities.gamepad.thumb_ly},
        {capabilities.gamepad.thumb_rx, capabilities.gamepad.thumb_ry}};
  }
}

std::vector<InputDriver*> InputSystem::FilterDrivers(uint32_t flags) {
  std::vector<InputDriver*> filtered_drivers;
  for (auto& driver : drivers_) {
    if (driver->GetInputType() == InputType::None) {
      continue;
    }

    if ((flags & driver->GetInputType()) != 0) {
      filtered_drivers.push_back(driver.get());
    }
  }
  return filtered_drivers;
}

X_RESULT InputSystem::GetCapabilities(uint32_t user_index, uint32_t flags,
                                      X_INPUT_CAPABILITIES* out_caps) {
  SCOPE_profile_cpu_f("hid");

  std::vector<InputDriver*> filtered_drivers = FilterDrivers(flags);

  for (auto& driver : filtered_drivers) {
    X_RESULT result = driver->GetCapabilities(user_index, flags, out_caps);
    if (result == X_ERROR_SUCCESS) {
      return result;
    }
  }
  // Report the scripted virtual controller as a standard gamepad so games that
  // gate on GetCapabilities treat slot 0 as connected.
  if ((!cvars::auto_input_script.empty() || !cvars::auto_input_file.empty()) &&
      user_index == 0 && out_caps) {
    std::memset(out_caps, 0, sizeof(*out_caps));
    out_caps->type = 0x01;      // XINPUT_DEVTYPE_GAMEPAD
    out_caps->sub_type = 0x01;  // XINPUT_DEVSUBTYPE_GAMEPAD
    out_caps->gamepad.buttons = 0xFFFF;
    out_caps->gamepad.left_trigger = 0xFF;
    out_caps->gamepad.right_trigger = 0xFF;
    out_caps->gamepad.thumb_lx = int16_t(0x7FFF);
    out_caps->gamepad.thumb_ly = int16_t(0x7FFF);
    out_caps->gamepad.thumb_rx = int16_t(0x7FFF);
    out_caps->gamepad.thumb_ry = int16_t(0x7FFF);
    out_caps->vibration.left_motor_speed = 0xFFFF;
    out_caps->vibration.right_motor_speed = 0xFFFF;
    return X_ERROR_SUCCESS;
  }
  return X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::GetState(uint32_t user_index, uint32_t flags,
                               X_INPUT_STATE* out_state) {
  SCOPE_profile_cpu_f("hid");

  std::vector<InputDriver*> filtered_drivers = FilterDrivers(flags);
  if (filtered_drivers.empty()) {
    if (MaybeApplyAutoInput(user_index, out_state)) {
      UpdateUsedSlot(nullptr, user_index, true);
      return X_ERROR_SUCCESS;
    }
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  for (auto& driver : filtered_drivers) {
    X_RESULT result = driver->GetState(user_index, out_state);
    if (result == X_ERROR_SUCCESS) {
      UpdateUsedSlot(driver, user_index, true);
      AdjustDeadzoneLevels(user_index, &out_state->gamepad);

      if (out_state->gamepad.buttons != 0) {
        last_used_slot = user_index;
      }
      if (!cvars::record_input.empty() && user_index == 0) {
        RecordInput(out_state->gamepad);
      }
      return result;
    }
  }
  if (MaybeApplyAutoInput(user_index, out_state)) {
    UpdateUsedSlot(nullptr, user_index, true);
    return X_ERROR_SUCCESS;
  }
  UpdateUsedSlot(nullptr, user_index, false);
  return X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::SetState(uint32_t user_index,
                               X_INPUT_VIBRATION* vibration) {
  SCOPE_profile_cpu_f("hid");
  X_INPUT_VIBRATION modified_vibration = ModifyVibrationLevel(vibration);
  for (auto& driver : drivers_) {
    X_RESULT result = driver->SetState(user_index, &modified_vibration);
    if (result == X_ERROR_SUCCESS) {
      return result;
    }
  }
  return X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::GetKeystroke(uint32_t user_index, uint32_t flags,
                                   X_INPUT_KEYSTROKE* out_keystroke) {
  SCOPE_profile_cpu_f("hid");

  std::vector<InputDriver*> filtered_drivers = FilterDrivers(flags);

  bool any_connected = false;
  for (auto& driver : filtered_drivers) {
    // connected_slots
    X_RESULT result = driver->GetKeystroke(user_index, flags, out_keystroke);
    if (result == X_ERROR_INVALID_PARAMETER ||
        result == X_ERROR_DEVICE_NOT_CONNECTED) {
      continue;
    }

    any_connected = true;

    if (result == X_ERROR_SUCCESS) {
      last_used_slot = user_index;
      return result;
    }

    if (result == X_ERROR_EMPTY) {
      continue;
    }
  }
  return any_connected ? X_ERROR_EMPTY : X_ERROR_DEVICE_NOT_CONNECTED;
}

bool InputSystem::GetVibrationCvar() { return cvars::vibration; }

void InputSystem::ToggleVibration() {
  OVERRIDE_bool(vibration, !cvars::vibration);
  // Send instant update to vibration state to prevent awaiting for next tick.
  X_INPUT_VIBRATION vibration = X_INPUT_VIBRATION();

  for (uint8_t user_index = 0; user_index < XUserMaxUserCount; user_index++) {
    SetState(user_index, &vibration);
  }
}

void InputSystem::AdjustDeadzoneLevels(const uint8_t slot,
                                       X_INPUT_GAMEPAD* gamepad) {
  if (slot > XUserMaxUserCount) {
    return;
  }

  // Left stick
  if (cvars::left_stick_deadzone_percentage > 0.0 &&
      cvars::left_stick_deadzone_percentage < 1.0) {
    const double deadzone_lx_percentage =
        controllers_max_joystick_value[slot].first.first *
        cvars::left_stick_deadzone_percentage;
    const double deadzone_ly_percentage =
        controllers_max_joystick_value[slot].first.second *
        cvars::left_stick_deadzone_percentage;

    const double theta = std::atan2(static_cast<double>(gamepad->thumb_ly),
                                    static_cast<double>(gamepad->thumb_lx));

    const double deadzone_y_value = std::sin(theta) * deadzone_ly_percentage;
    const double deadzone_x_value = std::cos(theta) * deadzone_lx_percentage;

    if (gamepad->thumb_ly > -deadzone_y_value &&
        gamepad->thumb_ly < deadzone_y_value) {
      gamepad->thumb_ly = 0;
    }

    if (gamepad->thumb_lx > -deadzone_x_value &&
        gamepad->thumb_lx < deadzone_x_value) {
      gamepad->thumb_lx = 0;
    }
  }

  // Right stick
  if (cvars::right_stick_deadzone_percentage > 0.0 &&
      cvars::right_stick_deadzone_percentage < 1.0) {
    const double deadzone_rx_percentage =
        controllers_max_joystick_value[slot].second.first *
        cvars::right_stick_deadzone_percentage;
    const double deadzone_ry_percentage =
        controllers_max_joystick_value[slot].second.second *
        cvars::right_stick_deadzone_percentage;

    const double theta = std::atan2(static_cast<double>(gamepad->thumb_ry),
                                    static_cast<double>(gamepad->thumb_rx));

    const double deadzone_y_value = std::sin(theta) * deadzone_ry_percentage;
    const double deadzone_x_value = std::cos(theta) * deadzone_rx_percentage;

    if (gamepad->thumb_ry > -deadzone_y_value &&
        gamepad->thumb_ry < deadzone_y_value) {
      gamepad->thumb_ry = 0;
    }

    if (gamepad->thumb_rx > -deadzone_x_value &&
        gamepad->thumb_rx < deadzone_x_value) {
      gamepad->thumb_rx = 0;
    }
  }
}

X_INPUT_VIBRATION InputSystem::ModifyVibrationLevel(
    X_INPUT_VIBRATION* vibration) {
  X_INPUT_VIBRATION modified_vibration = *vibration;
  if (cvars::vibration) {
    return modified_vibration;
  }

  // TODO(Gliniak): Use modifier instead of boolean value.
  modified_vibration.left_motor_speed = 0;
  modified_vibration.right_motor_speed = 0;
  return modified_vibration;
}
std::unique_lock<xe_unlikely_mutex> InputSystem::lock() {
  return std::unique_lock<xe_unlikely_mutex>{lock_};
}
}  // namespace hid
}  // namespace xe
