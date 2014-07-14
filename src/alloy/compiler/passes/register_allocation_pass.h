/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef ALLOY_COMPILER_PASSES_REGISTER_ALLOCATION_PASS_H_
#define ALLOY_COMPILER_PASSES_REGISTER_ALLOCATION_PASS_H_

#include <algorithm>
#include <bitset>
#include <vector>

#include <alloy/backend/machine_info.h>
#include <alloy/compiler/compiler_pass.h>

namespace alloy {
namespace compiler {
namespace passes {

class RegisterAllocationPass : public CompilerPass {
 public:
  RegisterAllocationPass(const backend::MachineInfo* machine_info);
  ~RegisterAllocationPass() override;

  int Run(hir::HIRBuilder* builder) override;

 private:
  // TODO(benvanik): rewrite all this set shit -- too much indirection, the
  // complexity is not needed.
  struct RegisterUsage {
    hir::Value* value;
    hir::Value::Use* use;
    RegisterUsage() : value(nullptr), use(nullptr) {}
    RegisterUsage(hir::Value* value_, hir::Value::Use* use_)
        : value(value_), use(use_) {}
    struct Comparer : std::binary_function<RegisterUsage, RegisterUsage, bool> {
      bool operator()(const RegisterUsage& a, const RegisterUsage& b) const {
        return a.use->instr->ordinal < b.use->instr->ordinal;
      }
    };
  };
  struct RegisterSetUsage {
    const backend::MachineInfo::RegisterSet* set = nullptr;
    uint32_t count = 0;
    std::bitset<32> availability = 0;
    // TODO(benvanik): another data type.
    std::vector<RegisterUsage> upcoming_uses;
  };

  void DumpUsage(const char* name);
  void PrepareBlockState();
  void AdvanceUses(hir::Instr* instr);
  bool IsRegInUse(const hir::RegAssignment& reg);
  RegisterSetUsage* MarkRegUsed(const hir::RegAssignment& reg,
                                hir::Value* value, hir::Value::Use* use);
  RegisterSetUsage* MarkRegAvailable(const hir::RegAssignment& reg);

  bool TryAllocateRegister(hir::Value* value,
                           const hir::RegAssignment& preferred_reg);
  bool TryAllocateRegister(hir::Value* value);
  bool SpillOneRegister(hir::HIRBuilder* builder, hir::TypeName required_type);

  RegisterSetUsage* RegisterSetForValue(const hir::Value* value);

  void SortUsageList(hir::Value* value);

 private:
  struct {
    RegisterSetUsage* int_set = nullptr;
    RegisterSetUsage* float_set = nullptr;
    RegisterSetUsage* vec_set = nullptr;
    RegisterSetUsage* all_sets[3];
  } usage_sets_;
};

}  // namespace passes
}  // namespace compiler
}  // namespace alloy

#endif  // ALLOY_COMPILER_PASSES_REGISTER_ALLOCATION_PASS_H_
