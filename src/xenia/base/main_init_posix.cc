/**
******************************************************************************
* Xenia : Xbox 360 Emulator Research Project                                 *
******************************************************************************
* Copyright 2023 Ben Vanik. All rights reserved.                             *
* Released under the BSD license - see LICENSE in the root for more details. *
******************************************************************************
*/
#include "xenia/base/platform.h"

#if !XE_PLATFORM_MAC
#include <xbyak/xbyak/xbyak_util.h>
#include <cstdio>
#include <cstdlib>

class StartupCpuFeatureCheck {
 public:
  StartupCpuFeatureCheck() {
    Xbyak::util::Cpu cpu;
    const char* error_message = nullptr;
    if (!cpu.has(Xbyak::util::Cpu::tAVX)) {
      error_message =
          "Your CPU does not support AVX, which is required by Xenia. See "
          "the "
          "FAQ for system requirements at https://xenia.jp";
    }
    if (error_message != nullptr) {
      fprintf(stderr, "WARNING: %s\n", error_message);
    }
  }
};

// This is a hack to get an instance of StartupAvxCheck
// constructed before any initialization code,
// where the AVX check then happens in the constructor.
// Ref:
// https://reviews.llvm.org/D12689#243295
__attribute__((
    init_priority(101))) static StartupCpuFeatureCheck gStartupAvxCheck;
#endif  // !XE_PLATFORM_MAC
