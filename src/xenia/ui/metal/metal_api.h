/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_METAL_METAL_API_H_
#define XENIA_UI_METAL_METAL_API_H_

// metal-cpp C++ wrappers for Metal and Foundation.
#include "third_party/metal-cpp/Metal/MTLDevice.hpp"
#include "third_party/metal-cpp/Metal/Metal.hpp"

#ifdef METAL_SHADER_CONVERTER_AVAILABLE
// IMPORTANT: Define this BEFORE including any Metal Shader Converter (MSC)
// headers. This tells MSC to use metal-cpp types instead of Objective-C.
#define IR_RUNTIME_METALCPP
// Include metal-cpp (without implementation - that's in metal_api.cc)
#include "third_party/metal-shader-converter/include/metal_irconverter/metal_irconverter.h"
// Don't include the runtime header here - that needs IR_PRIVATE_IMPLEMENTATION
// which should only be in ir_runtime_impl.cc
#endif  // METAL_SHADER_CONVERTER_AVAILABLE

#endif
