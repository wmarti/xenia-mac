/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/vulkan/spirv_tools_context.h"

#include <cstdlib>

#include "xenia/base/logging.h"
#include "xenia/base/platform.h"

#if XE_PLATFORM_WIN32 && XE_ARCH_ARM64
#define XE_DISABLE_SPIRV_TOOLS 1
#endif

#if !XE_DISABLE_SPIRV_TOOLS
#include <spirv-tools/optimizer.hpp>
#endif

namespace xe {
namespace ui {
namespace vulkan {

bool SpirvToolsContext::Initialize(unsigned int spirv_version) {
#if XE_DISABLE_SPIRV_TOOLS
  XELOGI("SPIRV-Tools: Disabled on Windows ARM64");
  return false;
#else
  // Determine target environment based on SPIR-V version
  if (spirv_version >= 0x10500) {
    target_env_ = SPV_ENV_VULKAN_1_2;
  } else if (spirv_version >= 0x10400) {
    target_env_ = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
  } else if (spirv_version >= 0x10300) {
    target_env_ = SPV_ENV_VULKAN_1_1;
  } else {
    target_env_ = SPV_ENV_VULKAN_1_0;
  }

  // Create SPIR-V context
  context_ = spvContextCreate(target_env_);
  if (!context_) {
    XELOGE("SPIRV-Tools: Failed to create context for target environment");
    return false;
  }

  XELOGI("SPIRV-Tools: Initialized successfully with static linking");
  return true;
#endif
}

void SpirvToolsContext::Shutdown() {
#if !XE_DISABLE_SPIRV_TOOLS
  if (context_) {
    spvContextDestroy(context_);
    context_ = nullptr;
  }
#endif
}

spv_result_t SpirvToolsContext::Validate(const uint32_t* words,
                                         size_t num_words,
                                         std::string* error) const {
#if XE_DISABLE_SPIRV_TOOLS
  return SPV_UNSUPPORTED;
#else
  if (error) {
    error->clear();
  }
  if (!context_) {
    return SPV_UNSUPPORTED;
  }

  // Create validator options with scalar block layout to match modern Vulkan
  // driver behavior. This is the most permissive layout option and matches
  // what VK_EXT_scalar_block_layout provides.
  spv_validator_options options = spvValidatorOptionsCreate();
  spvValidatorOptionsSetScalarBlockLayout(options, true);

  // Create binary struct for the validation API
  spv_const_binary_t binary = {words, num_words};

  spv_diagnostic diagnostic = nullptr;
  spv_result_t result =
      spvValidateWithOptions(context_, options, &binary, &diagnostic);

  spvValidatorOptionsDestroy(options);

  if (diagnostic) {
    if (error && diagnostic->error) {
      *error = diagnostic->error;
    }
    spvDiagnosticDestroy(diagnostic);
  }
  return result;
#endif
}

spv_result_t SpirvToolsContext::Optimize(const uint32_t* words,
                                         size_t num_words,
                                         std::vector<uint32_t>& optimized_words,
                                         bool performance_passes) {
#if XE_DISABLE_SPIRV_TOOLS
  return SPV_UNSUPPORTED;
#else
  optimized_words.clear();
  if (!context_) {
    return SPV_UNSUPPORTED;
  }

  // Use the C++ optimizer API for better integration
  spvtools::Optimizer optimizer(target_env_);

  // Set up message consumer for error reporting
  optimizer.SetMessageConsumer([](spv_message_level_t level, const char* source,
                                  const spv_position_t& position,
                                  const char* message) {
    if (level == SPV_MSG_ERROR || level == SPV_MSG_FATAL ||
        level == SPV_MSG_INTERNAL_ERROR) {
      XELOGE("SPIRV-Tools optimizer: {} {}", source ? source : "",
             message ? message : "");
    }
  });

  // Register optimization passes
  if (performance_passes) {
    optimizer.RegisterPerformancePasses();
  } else {
    optimizer.RegisterSizePasses();
  }

  // Run optimizer
  if (!optimizer.Run(words, num_words, &optimized_words)) {
    return SPV_ERROR_INVALID_BINARY;
  }

  return SPV_SUCCESS;
#endif
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe
