/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_MSL_SHADER_H_
#define XENIA_GPU_METAL_MSL_SHADER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "xenia/gpu/spirv_shader.h"
#include "xenia/ui/metal/metal_api.h"

namespace xe {
namespace gpu {
namespace metal {

// Metal shader translated via SPIR-V -> SPIRV-Cross -> MSL path.
// Inherits from SpirvShader (not DxbcShader), removing the MSC dependency.
class MslShader : public SpirvShader {
 public:
  MslShader(xenos::ShaderType shader_type, uint64_t ucode_data_hash,
            const uint32_t* ucode_dwords, size_t ucode_dword_count,
            std::endian ucode_source_endian = std::endian::big);

  // Per-modification translated shader for Metal.
  class MslTranslation : public SpirvTranslation {
   public:
    MslTranslation(MslShader& shader, uint64_t modification)
        : SpirvTranslation(shader, modification) {}
    ~MslTranslation();

    // Compile the SPIR-V binary (already in translated_binary()) to MSL
    // via SPIRV-Cross and create a Metal library + function.
    bool CompileToMsl(MTL::Device* device, bool is_ios = false);

    MTL::Library* metal_library() const { return metal_library_; }
    MTL::Function* metal_function() const { return metal_function_; }
    const std::string& msl_source() const { return msl_source_; }
    const std::string& entry_point_name() const { return entry_point_name_; }
    bool is_valid() const { return metal_function_ != nullptr; }

   private:
    MTL::Library* metal_library_ = nullptr;
    MTL::Function* metal_function_ = nullptr;
    std::string msl_source_;
    std::string entry_point_name_;
  };

 protected:
  Translation* CreateTranslationInstance(uint64_t modification) override;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_MSL_SHADER_H_
