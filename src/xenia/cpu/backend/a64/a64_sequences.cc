/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// A note about vectors:
// Xenia represents vectors as xyzw pairs, with indices 0123.
// XMM registers are xyzw pairs with indices 3210, making them more like wzyx.
// This makes things somewhat confusing. It'd be nice to just shuffle the
// registers around on load/store, however certain operations require that
// data be in the right offset.
// Basically, this identity must hold:
//   shuffle(vec, b00011011) -> {x,y,z,w} => {x,y,z,w}
// All indices and operations must respect that.
//
// Memory (big endian):
// [00 01 02 03] [04 05 06 07] [08 09 0A 0B] [0C 0D 0E 0F] (x, y, z, w)
// load into xmm register:
// [0F 0E 0D 0C] [0B 0A 09 08] [07 06 05 04] [03 02 01 00] (w, z, y, x)

#include "xenia/cpu/backend/a64/a64_sequences.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include "xenia/base/assert.h"
#include "xenia/base/clock.h"
#include "xenia/base/logging.h"
#include "xenia/base/string.h"
#include "xenia/base/threading.h"
#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/cpu/backend/a64/a64_emitter.h"
#include "xenia/cpu/backend/a64/a64_op.h"
#include "xenia/cpu/backend/a64/a64_tracers.h"
#include "xenia/cpu/backend/a64/a64_util.h"
#include "xenia/cpu/hir/hir_builder.h"
#include "xenia/cpu/processor.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

using namespace oaknut;

// TODO(benvanik): direct usings.
using namespace xe::cpu;
using namespace xe::cpu::hir;

using xe::cpu::hir::Instr;

typedef bool (*SequenceSelectFn)(A64Emitter&, const Instr*);

template <typename Fn>
void EmitWithVmxFpcr(A64Emitter& e, Fn&& emit_op) {
  // Altivec vector FP runs with non-IEEE flush-to-zero and round-to-nearest.
  e.MRS(X13, SystemReg::FPCR);
  e.MOV(X14, X13);
  e.BFI(X14, XZR, 22, 2);
  e.ORR(X14, X14, uint64_t(1) << 24);
  e.MSR(SystemReg::FPCR, X14);
  emit_op();
  e.MSR(SystemReg::FPCR, X13);
}

inline bool IsMulAddOrMulSubDef(const Value* value) {
  if (!value || !value->def || !value->def->opcode) {
    return false;
  }
  const auto opcode = value->def->opcode->num;
  return opcode == OPCODE_MUL_ADD || opcode == OPCODE_MUL_SUB;
}

inline void EmitPropagateInputNanF64Binary(A64Emitter& e, DReg dest, DReg src1,
                                           DReg src2, oaknut::Label& no_nan,
                                           oaknut::Label& done) {
  constexpr uint64_t kF64QuietBit = uint64_t(1) << 51;
  oaknut::Label check_src2;
  oaknut::Label src1_qnan;
  oaknut::Label src2_qnan;

  e.FCMP(src1, src1);
  e.B(Cond::VC, check_src2);
  e.FMOV(X0, src1);
  e.TST(X0, kF64QuietBit);
  e.B(Cond::NE, src1_qnan);
  e.ORR(X0, X0, kF64QuietBit);
  e.FMOV(dest, X0);
  e.B(done);
  e.l(src1_qnan);
  e.FMOV(dest, src1);
  e.B(done);

  e.l(check_src2);
  e.FCMP(src2, src2);
  e.B(Cond::VC, no_nan);
  e.FMOV(X0, src2);
  e.TST(X0, kF64QuietBit);
  e.B(Cond::NE, src2_qnan);
  e.ORR(X0, X0, kF64QuietBit);
  e.FMOV(dest, X0);
  e.B(done);
  e.l(src2_qnan);
  e.FMOV(dest, src2);
  e.B(done);
}

inline void EmitPropagateInputNanF64Ternary(A64Emitter& e, DReg dest, DReg src1,
                                            DReg src2, DReg src3,
                                            oaknut::Label& no_nan,
                                            oaknut::Label& done) {
  constexpr uint64_t kF64QuietBit = uint64_t(1) << 51;

  oaknut::Label check_src2;
  oaknut::Label check_src3;
  oaknut::Label src1_qnan;
  oaknut::Label src2_qnan;
  oaknut::Label src3_qnan;

  e.FCMP(src1, src1);
  e.B(Cond::VC, check_src2);
  e.FMOV(X0, src1);
  e.TST(X0, kF64QuietBit);
  e.B(Cond::NE, src1_qnan);
  e.ORR(X0, X0, kF64QuietBit);
  e.FMOV(dest, X0);
  e.B(done);
  e.l(src1_qnan);
  e.FMOV(dest, src1);
  e.B(done);

  e.l(check_src2);
  e.FCMP(src2, src2);
  e.B(Cond::VC, check_src3);
  e.FMOV(X0, src2);
  e.TST(X0, kF64QuietBit);
  e.B(Cond::NE, src2_qnan);
  e.ORR(X0, X0, kF64QuietBit);
  e.FMOV(dest, X0);
  e.B(done);
  e.l(src2_qnan);
  e.FMOV(dest, src2);
  e.B(done);

  e.l(check_src3);
  e.FCMP(src3, src3);
  e.B(Cond::VC, no_nan);
  e.FMOV(X0, src3);
  e.TST(X0, kF64QuietBit);
  e.B(Cond::NE, src3_qnan);
  e.ORR(X0, X0, kF64QuietBit);
  e.FMOV(dest, X0);
  e.B(done);
  e.l(src3_qnan);
  e.FMOV(dest, src3);
  e.B(done);
}

inline void EmitSelectFirstNanLanesF32x4(A64Emitter& e, QReg result, QReg src1,
                                         QReg src2) {
  // mask1 = isnan(src1)
  e.FCMEQ(Q3.S4(), src1.S4(), src1.S4());
  e.MVN(Q3.B16(), Q3.B16());
  e.BIT(result.B16(), src1.B16(), Q3.B16());

  // mask2 = !mask1 & isnan(src2)
  e.FCMEQ(Q0.S4(), src2.S4(), src2.S4());
  e.MVN(Q0.B16(), Q0.B16());
  e.BIC(Q0.B16(), Q0.B16(), Q3.B16());
  e.BIT(result.B16(), src2.B16(), Q0.B16());
}

inline void EmitSelectFirstQNanLanesF32x4(A64Emitter& e, QReg result, QReg src1,
                                          QReg src2, QReg src3) {
  // Prefer first NaN source lane and quiet signaling NaNs.
  e.LoadConstantV(Q6, vec128i(0x00400000u));

  e.FCMEQ(Q7.S4(), src1.S4(), src1.S4());
  e.MVN(Q7.B16(), Q7.B16());
  e.ORR(Q4.B16(), src1.B16(), Q6.B16());
  e.BIT(result.B16(), Q4.B16(), Q7.B16());

  e.FCMEQ(Q0.S4(), src2.S4(), src2.S4());
  e.MVN(Q0.B16(), Q0.B16());
  e.BIC(Q0.B16(), Q0.B16(), Q7.B16());
  e.ORR(Q4.B16(), src2.B16(), Q6.B16());
  e.BIT(result.B16(), Q4.B16(), Q0.B16());

  e.FCMEQ(Q3.S4(), src3.S4(), src3.S4());
  e.MVN(Q3.B16(), Q3.B16());
  e.ORR(Q4.B16(), src3.B16(), Q6.B16());
  e.BIC(Q3.B16(), Q3.B16(), Q7.B16());
  e.BIC(Q3.B16(), Q3.B16(), Q0.B16());
  e.BIT(result.B16(), Q4.B16(), Q3.B16());
}

// ============================================================================
// OPCODE_COMMENT
// ============================================================================
struct COMMENT : Sequence<COMMENT, I<OPCODE_COMMENT, VoidOp, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (IsTracingInstr()) {
      auto str = reinterpret_cast<const char*>(i.src1.value);
      // TODO(benvanik): pass through.
      // TODO(benvanik): don't just leak this memory.
      auto str_copy = xe_strdup(str);
      e.MOV(e.GetNativeParam(0), reinterpret_cast<uint64_t>(str_copy));
      e.CallNative(reinterpret_cast<void*>(TraceString));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_COMMENT, COMMENT);

// ============================================================================
// OPCODE_NOP
// ============================================================================
struct NOP : Sequence<NOP, I<OPCODE_NOP, VoidOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) { e.NOP(); }
};
EMITTER_OPCODE_TABLE(OPCODE_NOP, NOP);

// ============================================================================
// OPCODE_SOURCE_OFFSET
// ============================================================================
struct SOURCE_OFFSET
    : Sequence<SOURCE_OFFSET, I<OPCODE_SOURCE_OFFSET, VoidOp, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.MarkSourceOffset(i.instr);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SOURCE_OFFSET, SOURCE_OFFSET);

// ============================================================================
// OPCODE_ASSIGN
// ============================================================================
struct ASSIGN_I8 : Sequence<ASSIGN_I8, I<OPCODE_ASSIGN, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTB(i.dest, i.src1);
  }
};
struct ASSIGN_I16 : Sequence<ASSIGN_I16, I<OPCODE_ASSIGN, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTH(i.dest, i.src1);
  }
};
struct ASSIGN_I32 : Sequence<ASSIGN_I32, I<OPCODE_ASSIGN, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.MOV(i.dest, i.src1);
  }
};
struct ASSIGN_I64 : Sequence<ASSIGN_I64, I<OPCODE_ASSIGN, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.MOV(i.dest, i.src1);
  }
};
struct ASSIGN_F32 : Sequence<ASSIGN_F32, I<OPCODE_ASSIGN, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FMOV(i.dest, i.src1);
  }
};
struct ASSIGN_F64 : Sequence<ASSIGN_F64, I<OPCODE_ASSIGN, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FMOV(i.dest, i.src1);
  }
};
struct ASSIGN_V128 : Sequence<ASSIGN_V128, I<OPCODE_ASSIGN, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.LoadConstantV(i.dest.reg(), i.src1.constant());
    } else {
      e.MOV(i.dest.reg().B16(), i.src1.reg().B16());
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ASSIGN, ASSIGN_I8, ASSIGN_I16, ASSIGN_I32,
                     ASSIGN_I64, ASSIGN_F32, ASSIGN_F64, ASSIGN_V128);

// ============================================================================
// OPCODE_CAST
// ============================================================================
struct CAST_I32_F32 : Sequence<CAST_I32_F32, I<OPCODE_CAST, I32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FMOV(i.dest, i.src1);
  }
};
struct CAST_I64_F64 : Sequence<CAST_I64_F64, I<OPCODE_CAST, I64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FMOV(i.dest, i.src1);
  }
};
struct CAST_F32_I32 : Sequence<CAST_F32_I32, I<OPCODE_CAST, F32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FMOV(i.dest, i.src1);
  }
};
struct CAST_F64_I64 : Sequence<CAST_F64_I64, I<OPCODE_CAST, F64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FMOV(i.dest, i.src1);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CAST, CAST_I32_F32, CAST_I64_F64, CAST_F32_I32,
                     CAST_F64_I64);

// ============================================================================
// OPCODE_ZERO_EXTEND
// ============================================================================
struct ZERO_EXTEND_I16_I8
    : Sequence<ZERO_EXTEND_I16_I8, I<OPCODE_ZERO_EXTEND, I16Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTB(i.dest, i.src1);
  }
};
struct ZERO_EXTEND_I32_I8
    : Sequence<ZERO_EXTEND_I32_I8, I<OPCODE_ZERO_EXTEND, I32Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTB(i.dest, i.src1);
  }
};
struct ZERO_EXTEND_I64_I8
    : Sequence<ZERO_EXTEND_I64_I8, I<OPCODE_ZERO_EXTEND, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTB(i.dest.reg().toW(), i.src1);
  }
};
struct ZERO_EXTEND_I32_I16
    : Sequence<ZERO_EXTEND_I32_I16, I<OPCODE_ZERO_EXTEND, I32Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTH(i.dest, i.src1);
  }
};
struct ZERO_EXTEND_I64_I16
    : Sequence<ZERO_EXTEND_I64_I16, I<OPCODE_ZERO_EXTEND, I64Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTH(i.dest.reg().toW(), i.src1);
  }
};
struct ZERO_EXTEND_I64_I32
    : Sequence<ZERO_EXTEND_I64_I32, I<OPCODE_ZERO_EXTEND, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.MOV(i.dest.reg().toW(), i.src1);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ZERO_EXTEND, ZERO_EXTEND_I16_I8, ZERO_EXTEND_I32_I8,
                     ZERO_EXTEND_I64_I8, ZERO_EXTEND_I32_I16,
                     ZERO_EXTEND_I64_I16, ZERO_EXTEND_I64_I32);

// ============================================================================
// OPCODE_SIGN_EXTEND
// ============================================================================
struct SIGN_EXTEND_I16_I8
    : Sequence<SIGN_EXTEND_I16_I8, I<OPCODE_SIGN_EXTEND, I16Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.SXTB(i.dest, i.src1);
  }
};
struct SIGN_EXTEND_I32_I8
    : Sequence<SIGN_EXTEND_I32_I8, I<OPCODE_SIGN_EXTEND, I32Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.SXTB(i.dest, i.src1);
  }
};
struct SIGN_EXTEND_I64_I8
    : Sequence<SIGN_EXTEND_I64_I8, I<OPCODE_SIGN_EXTEND, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.SXTB(i.dest, i.src1);
  }
};
struct SIGN_EXTEND_I32_I16
    : Sequence<SIGN_EXTEND_I32_I16, I<OPCODE_SIGN_EXTEND, I32Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.SXTH(i.dest, i.src1);
  }
};
struct SIGN_EXTEND_I64_I16
    : Sequence<SIGN_EXTEND_I64_I16, I<OPCODE_SIGN_EXTEND, I64Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.SXTH(i.dest, i.src1);
  }
};
struct SIGN_EXTEND_I64_I32
    : Sequence<SIGN_EXTEND_I64_I32, I<OPCODE_SIGN_EXTEND, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.SXTW(i.dest, i.src1.reg());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SIGN_EXTEND, SIGN_EXTEND_I16_I8, SIGN_EXTEND_I32_I8,
                     SIGN_EXTEND_I64_I8, SIGN_EXTEND_I32_I16,
                     SIGN_EXTEND_I64_I16, SIGN_EXTEND_I64_I32);

// ============================================================================
// OPCODE_TRUNCATE
// ============================================================================
struct TRUNCATE_I8_I16
    : Sequence<TRUNCATE_I8_I16, I<OPCODE_TRUNCATE, I8Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTB(i.dest, i.src1);
  }
};
struct TRUNCATE_I8_I32
    : Sequence<TRUNCATE_I8_I32, I<OPCODE_TRUNCATE, I8Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTB(i.dest, i.src1);
  }
};
struct TRUNCATE_I8_I64
    : Sequence<TRUNCATE_I8_I64, I<OPCODE_TRUNCATE, I8Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTB(i.dest, i.src1.reg().toW());
  }
};
struct TRUNCATE_I16_I32
    : Sequence<TRUNCATE_I16_I32, I<OPCODE_TRUNCATE, I16Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTH(i.dest, i.src1);
  }
};
struct TRUNCATE_I16_I64
    : Sequence<TRUNCATE_I16_I64, I<OPCODE_TRUNCATE, I16Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.UXTH(i.dest, i.src1.reg().toW());
  }
};
struct TRUNCATE_I32_I64
    : Sequence<TRUNCATE_I32_I64, I<OPCODE_TRUNCATE, I32Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.MOV(i.dest, i.src1.reg().toW());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_TRUNCATE, TRUNCATE_I8_I16, TRUNCATE_I8_I32,
                     TRUNCATE_I8_I64, TRUNCATE_I16_I32, TRUNCATE_I16_I64,
                     TRUNCATE_I32_I64);

// ============================================================================
// OPCODE_CONVERT
// ============================================================================
struct CONVERT_I32_F32
    : Sequence<CONVERT_I32_F32, I<OPCODE_CONVERT, I32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // TODO(benvanik): saturation check? cvtt* (trunc?)
    const SReg src = i.src1.is_constant ? S0 : i.src1.reg().toS();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    if (i.instr->flags == ROUND_TO_ZERO) {
      e.FCVTZS(i.dest, src);
    } else {
      e.FCVTNS(i.dest, src);
    }
  }
};
struct CONVERT_I32_F64
    : Sequence<CONVERT_I32_F64, I<OPCODE_CONVERT, I32Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // Intel returns 0x80000000 if the double value does not fit within an int32
    // ARM64 and PPC saturates the value instead
    const DReg src = i.src1.is_constant ? D0 : i.src1.reg().toD();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    if (i.instr->flags == ROUND_TO_ZERO) {
      e.FCVTZS(i.dest, src);
    } else {
      e.FCVTNS(i.dest, src);
    }
  }
};
struct CONVERT_I64_F64
    : Sequence<CONVERT_I64_F64, I<OPCODE_CONVERT, I64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const DReg src = i.src1.is_constant ? D0 : i.src1.reg().toD();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    if (i.instr->flags == ROUND_TO_ZERO) {
      e.FCVTZS(i.dest, src);
    } else {
      e.FCVTNS(i.dest, src);
    }
  }
};
struct CONVERT_F32_I32
    : Sequence<CONVERT_F32_I32, I<OPCODE_CONVERT, F32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.SCVTF(i.dest.reg().toS(), i.src1);
  }
};
struct CONVERT_F32_F64
    : Sequence<CONVERT_F32_F64, I<OPCODE_CONVERT, F32Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const DReg src = i.src1.is_constant ? D0 : i.src1;
    if (i.src1.is_constant) {
      e.LoadConstantV(src.toQ(), i.src1.constant());
    }
    e.FCVT(i.dest.reg().toS(), src.toD());
  }
};
struct CONVERT_F64_I64
    : Sequence<CONVERT_F64_I64, I<OPCODE_CONVERT, F64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.SCVTF(i.dest.reg().toD(), i.src1);
  }
};
struct CONVERT_F64_F32
    : Sequence<CONVERT_F64_F32, I<OPCODE_CONVERT, F64Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const SReg src = i.src1.is_constant ? S0 : i.src1;
    if (i.src1.is_constant) {
      e.LoadConstantV(src.toQ(), i.src1.constant());
    }
    e.FCVT(i.dest.reg().toD(), src.toS());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CONVERT, CONVERT_I32_F32, CONVERT_I32_F64,
                     CONVERT_I64_F64, CONVERT_F32_I32, CONVERT_F32_F64,
                     CONVERT_F64_I64, CONVERT_F64_F32);

// ============================================================================
// OPCODE_TO_SINGLE
// ============================================================================
struct TOSINGLE_F64_F64
    : Sequence<TOSINGLE_F64_F64, I<OPCODE_TO_SINGLE, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const DReg src = i.src1.is_constant ? D1 : i.src1;
    if (i.src1.is_constant) {
      e.LoadConstantV(src.toQ(), i.src1.constant());
    }
    e.FCVT(S0, src);
    e.FCVT(i.dest, S0);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_TO_SINGLE, TOSINGLE_F64_F64);

// ============================================================================
// OPCODE_ROUND
// ============================================================================
struct ROUND_F32 : Sequence<ROUND_F32, I<OPCODE_ROUND, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const SReg src = i.src1.is_constant ? S0 : i.src1.reg().toS();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    switch (i.instr->flags) {
      case ROUND_TO_ZERO:
        e.FRINTZ(i.dest.reg().toS(), src);
        break;
      case ROUND_TO_NEAREST:
        e.FRINTN(i.dest.reg().toS(), src);
        break;
      case ROUND_TO_MINUS_INFINITY:
        e.FRINTM(i.dest.reg().toS(), src);
        break;
      case ROUND_TO_POSITIVE_INFINITY:
        e.FRINTP(i.dest.reg().toS(), src);
        break;
    }
  }
};
struct ROUND_F64 : Sequence<ROUND_F64, I<OPCODE_ROUND, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const DReg src = i.src1.is_constant ? D0 : i.src1;
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    switch (i.instr->flags) {
      case ROUND_TO_ZERO:
        e.FRINTZ(i.dest, src);
        break;
      case ROUND_TO_NEAREST:
        e.FRINTN(i.dest, src);
        break;
      case ROUND_TO_MINUS_INFINITY:
        e.FRINTM(i.dest, src);
        break;
      case ROUND_TO_POSITIVE_INFINITY:
        e.FRINTP(i.dest, src);
        break;
    }
  }
};
struct ROUND_V128 : Sequence<ROUND_V128, I<OPCODE_ROUND, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    QReg src = i.src1.is_constant ? Q0 : i.src1.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    EmitWithVmxFpcr(e, [&] {
      switch (i.instr->flags) {
        case ROUND_TO_ZERO:
          e.FRINTZ(i.dest.reg().S4(), src.S4());
          break;
        case ROUND_TO_NEAREST:
          e.FRINTN(i.dest.reg().S4(), src.S4());
          break;
        case ROUND_TO_MINUS_INFINITY:
          e.FRINTM(i.dest.reg().S4(), src.S4());
          break;
        case ROUND_TO_POSITIVE_INFINITY:
          e.FRINTP(i.dest.reg().S4(), src.S4());
          break;
      }
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ROUND, ROUND_F32, ROUND_F64, ROUND_V128);

// ============================================================================
// OPCODE_LOAD_CLOCK
// ============================================================================
struct LOAD_CLOCK : Sequence<LOAD_CLOCK, I<OPCODE_LOAD_CLOCK, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // When scaling is disabled and the raw clock source is selected, the code
    // in the Clock class is actually just forwarding tick counts after one
    // simple multiply and division. In that case we rather bake the scaling in
    // here to cut extra function calls with CPU cache misses and stack frame
    // overhead.
    if (cvars::clock_no_scaling && cvars::clock_source_raw) {
      auto ratio = Clock::guest_tick_ratio();
      // The 360 CPU is an in-order CPU, ARM64 usually isn't. Since it's
      // resolution however is much higher than the 360's mftb instruction this
      // can safely be ignored.

      // Read clock cycle count
      e.MRS(i.dest, SystemReg::CNTVCT_EL0);
      // Apply tick frequency scaling.
      e.MOV(X0, ratio.first);
      e.MUL(i.dest, i.dest, X0);
      e.MOV(X0, ratio.second);
      e.UDIV(i.dest, i.dest, X0);
    } else {
      e.CallNative(LoadClock);
      e.MOV(i.dest, X0);
    }
  }
  static uint64_t LoadClock(void* raw_context) {
    return Clock::QueryGuestTickCount();
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_CLOCK, LOAD_CLOCK);

// ============================================================================
// OPCODE_CONTEXT_BARRIER
// ============================================================================
struct CONTEXT_BARRIER
    : Sequence<CONTEXT_BARRIER, I<OPCODE_CONTEXT_BARRIER, VoidOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {}
};
EMITTER_OPCODE_TABLE(OPCODE_CONTEXT_BARRIER, CONTEXT_BARRIER);

// ============================================================================
// OPCODE_MAX
// ============================================================================
struct MAX_F32 : Sequence<MAX_F32, I<OPCODE_MAX, F32Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp<SReg>(
        e, i, [](A64Emitter& e, SReg dest, SReg src1, SReg src2) {
          e.FMAX(dest, src1, src2);
        });
  }
};
struct MAX_F64 : Sequence<MAX_F64, I<OPCODE_MAX, F64Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp<DReg>(
        e, i, [](A64Emitter& e, DReg dest, DReg src1, DReg src2) {
          e.FMAX(dest, src1, src2);
        });
  }
};
struct MAX_V128 : Sequence<MAX_V128, I<OPCODE_MAX, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitWithVmxFpcr(e, [&] {
      QReg src1 = Q0;
      QReg src2 = Q1;
      if (i.src1.is_constant) {
        e.LoadConstantV(src1, i.src1.constant());
      } else {
        e.MOV(src1.B16(), i.src1.reg().B16());
      }
      if (i.src2.is_constant) {
        e.LoadConstantV(src2, i.src2.constant());
      } else {
        e.MOV(src2.B16(), i.src2.reg().B16());
      }

      e.FMAX(Q2.S4(), src1.S4(), src2.S4());
      EmitSelectFirstNanLanesF32x4(e, Q2, src1, src2);
      e.MOV(i.dest.reg().B16(), Q2.B16());
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MAX, MAX_F32, MAX_F64, MAX_V128);

// ============================================================================
// OPCODE_MIN
// ============================================================================
struct MIN_I8 : Sequence<MIN_I8, I<OPCODE_MIN, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.CMP(dest_src, src);
          e.CSEL(dest_src, dest_src, src, Cond::LO);
        },
        [](A64Emitter& e, WReg dest_src, int32_t constant) {
          e.MOV(W0, constant);
          e.CMP(dest_src, W0);
          e.CSEL(dest_src, dest_src, W0, Cond::LO);
        });
  }
};
struct MIN_I16 : Sequence<MIN_I16, I<OPCODE_MIN, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.CMP(dest_src, src);
          e.CSEL(dest_src, dest_src, src, Cond::LO);
        },
        [](A64Emitter& e, WReg dest_src, int32_t constant) {
          e.MOV(W0, constant);
          e.CMP(dest_src, W0);
          e.CSEL(dest_src, dest_src, W0, Cond::LO);
        });
  }
};
struct MIN_I32 : Sequence<MIN_I32, I<OPCODE_MIN, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.CMP(dest_src, src);
          e.CSEL(dest_src, dest_src, src, Cond::LO);
        },
        [](A64Emitter& e, WReg dest_src, int32_t constant) {
          e.MOV(W0, constant);
          e.CMP(dest_src, W0);
          e.CSEL(dest_src, dest_src, W0, Cond::LO);
        });
  }
};
struct MIN_I64 : Sequence<MIN_I64, I<OPCODE_MIN, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryOp(
        e, i,
        [](A64Emitter& e, XReg dest_src, XReg src) {
          e.CMP(dest_src, src);
          e.CSEL(dest_src, dest_src, src, Cond::LO);
        },
        [](A64Emitter& e, XReg dest_src, int64_t constant) {
          e.MOV(X0, constant);
          e.CMP(dest_src, X0);
          e.CSEL(dest_src, dest_src, X0, Cond::LO);
        });
  }
};
struct MIN_F32 : Sequence<MIN_F32, I<OPCODE_MIN, F32Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp<SReg>(
        e, i, [](A64Emitter& e, SReg dest, SReg src1, SReg src2) {
          e.FMIN(dest, src1, src2);
        });
  }
};
struct MIN_F64 : Sequence<MIN_F64, I<OPCODE_MIN, F64Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp<DReg>(
        e, i, [](A64Emitter& e, DReg dest, DReg src1, DReg src2) {
          e.FMIN(dest, src1, src2);
        });
  }
};
struct MIN_V128 : Sequence<MIN_V128, I<OPCODE_MIN, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitWithVmxFpcr(e, [&] {
      QReg src1 = Q0;
      QReg src2 = Q1;
      if (i.src1.is_constant) {
        e.LoadConstantV(src1, i.src1.constant());
      } else {
        e.MOV(src1.B16(), i.src1.reg().B16());
      }
      if (i.src2.is_constant) {
        e.LoadConstantV(src2, i.src2.constant());
      } else {
        e.MOV(src2.B16(), i.src2.reg().B16());
      }

      e.FMIN(Q2.S4(), src1.S4(), src2.S4());
      EmitSelectFirstNanLanesF32x4(e, Q2, src1, src2);
      e.MOV(i.dest.reg().B16(), Q2.B16());
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MIN, MIN_I8, MIN_I16, MIN_I32, MIN_I64, MIN_F32,
                     MIN_F64, MIN_V128);

// ============================================================================
// OPCODE_SELECT
// ============================================================================
// dest = src1 ? src2 : src3
// TODO(benvanik): match compare + select sequences, as often it's something
//     like SELECT(VECTOR_COMPARE_SGE(a, b), a, b)
struct SELECT_I8
    : Sequence<SELECT_I8, I<OPCODE_SELECT, I8Op, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    WReg src2(0);
    if (i.src2.is_constant) {
      src2 = W0;
      e.MOV(src2, i.src2.constant());
    } else {
      src2 = i.src2;
    }
    e.CMP(i.src1.reg().toX(), 0);
    e.CSEL(i.dest, src2, i.src3, Cond::NE);
  }
};
struct SELECT_I16
    : Sequence<SELECT_I16, I<OPCODE_SELECT, I16Op, I8Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    WReg src2(0);
    if (i.src2.is_constant) {
      src2 = W0;
      e.MOV(src2, i.src2.constant());
    } else {
      src2 = i.src2;
    }
    e.CMP(i.src1.reg().toX(), 0);
    e.CSEL(i.dest, src2, i.src3, Cond::NE);
  }
};
struct SELECT_I32
    : Sequence<SELECT_I32, I<OPCODE_SELECT, I32Op, I8Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    WReg src2(0);
    if (i.src2.is_constant) {
      src2 = W0;
      e.MOV(src2, i.src2.constant());
    } else {
      src2 = i.src2;
    }
    e.CMP(i.src1.reg().toX(), 0);
    e.CSEL(i.dest, src2, i.src3, Cond::NE);
  }
};
struct SELECT_I64
    : Sequence<SELECT_I64, I<OPCODE_SELECT, I64Op, I8Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    XReg src2(0);
    if (i.src2.is_constant) {
      src2 = X0;
      e.MOV(src2, i.src2.constant());
    } else {
      src2 = i.src2;
    }
    e.CMP(i.src1.reg().toX(), 0);
    e.CSEL(i.dest, src2, i.src3, Cond::NE);
  }
};
struct SELECT_F32
    : Sequence<SELECT_F32, I<OPCODE_SELECT, F32Op, I8Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // dest = src1 != 0 ? src2 : src3

    SReg src2 = i.src2.is_constant ? S2 : i.src2;
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    }

    SReg src3 = i.src3.is_constant ? S3 : i.src3;
    if (i.src3.is_constant) {
      e.LoadConstantV(src3.toQ(), i.src3.constant());
    }

    e.CMP(i.src1.reg().toX(), 0);
    e.FCSEL(i.dest, src2, src3, Cond::NE);
  }
};
struct SELECT_F64
    : Sequence<SELECT_F64, I<OPCODE_SELECT, F64Op, I8Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // dest = src1 != 0 ? src2 : src3

    const DReg src2 = i.src2.is_constant ? D2 : i.src2;
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    }

    const DReg src3 = i.src3.is_constant ? D3 : i.src3;
    if (i.src3.is_constant) {
      e.LoadConstantV(src3.toQ(), i.src3.constant());
    }

    e.CMP(i.src1.reg().toX(), 0);
    e.FCSEL(i.dest, src2, src3, Cond::NE);
  }
};
struct SELECT_V128_I8
    : Sequence<SELECT_V128_I8, I<OPCODE_SELECT, V128Op, I8Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // dest = src1 != 0 ? src2 : src3

    const QReg src2 = i.src2.is_constant ? Q2 : i.src2;
    if (i.src2.is_constant) {
      e.LoadConstantV(src2, i.src2.constant());
    }

    const QReg src3 = i.src3.is_constant ? Q3 : i.src3;
    if (i.src3.is_constant) {
      e.LoadConstantV(src3, i.src3.constant());
    }

    e.CMP(i.src1.reg().toX(), 0);
    e.CSETM(W0, Cond::NE);
    e.DUP(i.dest.reg().S4(), W0);
    e.BSL(i.dest.reg().B16(), src2.B16(), src3.B16());
  }
};
struct SELECT_V128_V128
    : Sequence<SELECT_V128_V128,
               I<OPCODE_SELECT, V128Op, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const QReg src1 = Q0;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1, i.src1.constant());
    } else {
      e.MOV(src1.B16(), i.src1.reg().B16());
    }

    const QReg src2 = i.src2.is_constant ? Q2 : i.src2;
    if (i.src2.is_constant) {
      e.LoadConstantV(src2, i.src2.constant());
    }

    const QReg src3 = i.src3.is_constant ? Q3 : i.src3;
    if (i.src3.is_constant) {
      e.LoadConstantV(src3, i.src3.constant());
    }

    // src1 ? src2 : src3;
    e.BSL(src1.B16(), src3.B16(), src2.B16());
    e.MOV(i.dest.reg().B16(), src1.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SELECT, SELECT_I8, SELECT_I16, SELECT_I32,
                     SELECT_I64, SELECT_F32, SELECT_F64, SELECT_V128_I8,
                     SELECT_V128_V128);

// ============================================================================
// OPCODE_IS_NAN
// ============================================================================
struct IS_NAN_F32 : Sequence<IS_NAN_F32, I<OPCODE_IS_NAN, I8Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FCMP(i.src1, i.src1);
    e.CSET(i.dest, Cond::VS);
  }
};

struct IS_NAN_F64 : Sequence<IS_NAN_F64, I<OPCODE_IS_NAN, I8Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FCMP(i.src1, i.src1);
    e.CSET(i.dest, Cond::VS);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_IS_NAN, IS_NAN_F32, IS_NAN_F64);

// ============================================================================
// OPCODE_COMPARE_EQ
// ============================================================================
struct COMPARE_EQ_I8
    : Sequence<COMPARE_EQ_I8, I<OPCODE_COMPARE_EQ, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeCompareOp(
        e, i, [](A64Emitter& e, WReg src1, WReg src2) { e.CMP(src1, src2); },
        [](A64Emitter& e, WReg src1, int32_t constant) {
          e.MOV(W1, constant);
          e.CMP(src1, W1);
        });
    e.CSET(i.dest, Cond::EQ);
  }
};
struct COMPARE_EQ_I16
    : Sequence<COMPARE_EQ_I16, I<OPCODE_COMPARE_EQ, I8Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeCompareOp(
        e, i, [](A64Emitter& e, WReg src1, WReg src2) { e.CMP(src1, src2); },
        [](A64Emitter& e, WReg src1, int32_t constant) {
          e.MOV(W1, constant);
          e.CMP(src1, W1);
        });
    e.CSET(i.dest, Cond::EQ);
  }
};
struct COMPARE_EQ_I32
    : Sequence<COMPARE_EQ_I32, I<OPCODE_COMPARE_EQ, I8Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeCompareOp(
        e, i, [](A64Emitter& e, WReg src1, WReg src2) { e.CMP(src1, src2); },
        [](A64Emitter& e, WReg src1, int32_t constant) {
          e.MOV(W1, constant);
          e.CMP(src1, W1);
        });
    e.CSET(i.dest, Cond::EQ);
  }
};
struct COMPARE_EQ_I64
    : Sequence<COMPARE_EQ_I64, I<OPCODE_COMPARE_EQ, I8Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeCompareOp(
        e, i, [](A64Emitter& e, XReg src1, XReg src2) { e.CMP(src1, src2); },
        [](A64Emitter& e, XReg src1, int32_t constant) {
          e.MOV(X1, constant);
          e.CMP(src1, X1);
        });
    e.CSET(i.dest, Cond::EQ);
  }
};
struct COMPARE_EQ_F32
    : Sequence<COMPARE_EQ_F32, I<OPCODE_COMPARE_EQ, I8Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp<SReg>(
        e, i, [](A64Emitter& e, I8Op dest, const SReg& src1, const SReg& src2) {
          e.FCMP(src1, src2);
        });
    e.CSET(i.dest, Cond::EQ);
  }
};
struct COMPARE_EQ_F64
    : Sequence<COMPARE_EQ_F64, I<OPCODE_COMPARE_EQ, I8Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp<DReg>(
        e, i, [](A64Emitter& e, I8Op dest, const DReg& src1, const DReg& src2) {
          e.FCMP(src1, src2);
        });
    e.CSET(i.dest, Cond::EQ);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_COMPARE_EQ, COMPARE_EQ_I8, COMPARE_EQ_I16,
                     COMPARE_EQ_I32, COMPARE_EQ_I64, COMPARE_EQ_F32,
                     COMPARE_EQ_F64);

// ============================================================================
// OPCODE_COMPARE_NE
// ============================================================================
struct COMPARE_NE_I8
    : Sequence<COMPARE_NE_I8, I<OPCODE_COMPARE_NE, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeCompareOp(
        e, i, [](A64Emitter& e, WReg src1, WReg src2) { e.CMP(src1, src2); },
        [](A64Emitter& e, WReg src1, int32_t constant) {
          e.MOV(W1, constant);
          e.CMP(src1, W1);
        });
    e.CSET(i.dest, Cond::NE);
  }
};
struct COMPARE_NE_I16
    : Sequence<COMPARE_NE_I16, I<OPCODE_COMPARE_NE, I8Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeCompareOp(
        e, i, [](A64Emitter& e, WReg src1, WReg src2) { e.CMP(src1, src2); },
        [](A64Emitter& e, WReg src1, int32_t constant) {
          e.MOV(W1, constant);
          e.CMP(src1, W1);
        });
    e.CSET(i.dest, Cond::NE);
  }
};
struct COMPARE_NE_I32
    : Sequence<COMPARE_NE_I32, I<OPCODE_COMPARE_NE, I8Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeCompareOp(
        e, i, [](A64Emitter& e, WReg src1, WReg src2) { e.CMP(src1, src2); },
        [](A64Emitter& e, WReg src1, int32_t constant) {
          e.MOV(W1, constant);
          e.CMP(src1, W1);
        });
    e.CSET(i.dest, Cond::NE);
  }
};
struct COMPARE_NE_I64
    : Sequence<COMPARE_NE_I64, I<OPCODE_COMPARE_NE, I8Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeCompareOp(
        e, i, [](A64Emitter& e, XReg src1, XReg src2) { e.CMP(src1, src2); },
        [](A64Emitter& e, XReg src1, int32_t constant) {
          e.MOV(X1, constant);
          e.CMP(src1, X1);
        });
    e.CSET(i.dest, Cond::NE);
  }
};
struct COMPARE_NE_F32
    : Sequence<COMPARE_NE_F32, I<OPCODE_COMPARE_NE, I8Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FCMP(i.src1, i.src2);
    e.CSET(i.dest, Cond::NE);
  }
};
struct COMPARE_NE_F64
    : Sequence<COMPARE_NE_F64, I<OPCODE_COMPARE_NE, I8Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FCMP(i.src1, i.src2);
    e.CSET(i.dest, Cond::NE);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_COMPARE_NE, COMPARE_NE_I8, COMPARE_NE_I16,
                     COMPARE_NE_I32, COMPARE_NE_I64, COMPARE_NE_F32,
                     COMPARE_NE_F64);

// ============================================================================
// OPCODE_COMPARE_*
// ============================================================================
#define EMITTER_ASSOCIATIVE_COMPARE_INT(op, cond, inverse_cond, type,          \
                                        reg_type)                              \
  struct COMPARE_##op##_##type                                                 \
      : Sequence<COMPARE_##op##_##type,                                        \
                 I<OPCODE_COMPARE_##op, I8Op, type, type>> {                   \
    static void Emit(A64Emitter& e, const EmitArgType& i) {                    \
      EmitAssociativeCompareOp(                                                \
          e, i,                                                                \
          [](A64Emitter& e, WReg dest, const reg_type& src1,                   \
             const reg_type& src2, bool inverse) {                             \
            e.CMP(src1, src2);                                                 \
            if (!inverse) {                                                    \
              e.CSET(dest, cond);                                              \
            } else {                                                           \
              e.CSET(dest, inverse_cond);                                      \
            }                                                                  \
          },                                                                   \
          [](A64Emitter& e, WReg dest, const reg_type& src1, int32_t constant, \
             bool inverse) {                                                   \
            e.MOV(reg_type(1), constant);                                      \
            e.CMP(src1, reg_type(1));                                          \
            if (!inverse) {                                                    \
              e.CSET(dest, cond);                                              \
            } else {                                                           \
              e.CSET(dest, inverse_cond);                                      \
            }                                                                  \
          });                                                                  \
    }                                                                          \
  };
#define EMITTER_ASSOCIATIVE_COMPARE_XX(op, cond, inverse_cond)          \
  EMITTER_ASSOCIATIVE_COMPARE_INT(op, cond, inverse_cond, I8Op, WReg);  \
  EMITTER_ASSOCIATIVE_COMPARE_INT(op, cond, inverse_cond, I16Op, WReg); \
  EMITTER_ASSOCIATIVE_COMPARE_INT(op, cond, inverse_cond, I32Op, WReg); \
  EMITTER_ASSOCIATIVE_COMPARE_INT(op, cond, inverse_cond, I64Op, XReg); \
  EMITTER_OPCODE_TABLE(OPCODE_COMPARE_##op, COMPARE_##op##_I8Op,        \
                       COMPARE_##op##_I16Op, COMPARE_##op##_I32Op,      \
                       COMPARE_##op##_I64Op);
EMITTER_ASSOCIATIVE_COMPARE_XX(SLT, Cond::LT, Cond::GT);  // setl, setg
EMITTER_ASSOCIATIVE_COMPARE_XX(SLE, Cond::LE, Cond::GE);  // setle, setge
EMITTER_ASSOCIATIVE_COMPARE_XX(SGT, Cond::GT, Cond::LT);  // setg, setl
EMITTER_ASSOCIATIVE_COMPARE_XX(SGE, Cond::GE, Cond::LE);  // setge, setle
EMITTER_ASSOCIATIVE_COMPARE_XX(ULT, Cond::LO, Cond::HI);  // setb, seta
EMITTER_ASSOCIATIVE_COMPARE_XX(ULE, Cond::LS, Cond::HS);  // setbe, setae
EMITTER_ASSOCIATIVE_COMPARE_XX(UGE, Cond::HS, Cond::LS);  // setae, setbe
EMITTER_ASSOCIATIVE_COMPARE_XX(UGT, Cond::HI, Cond::LO);  // seta, setb

// https://web.archive.org/web/20171129015931/https://x86.renejeschke.de/html/file_module_x86_id_288.html
// Original link: https://x86.renejeschke.de/html/file_module_x86_id_288.html
#define EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(op, cond)                  \
  struct COMPARE_##op##_F32                                           \
      : Sequence<COMPARE_##op##_F32,                                  \
                 I<OPCODE_COMPARE_##op, I8Op, F32Op, F32Op>> {        \
    static void Emit(A64Emitter& e, const EmitArgType& i) {           \
      e.FCMP(i.src1, i.src2);                                         \
      e.CSET(i.dest, cond);                                           \
    }                                                                 \
  };                                                                  \
  struct COMPARE_##op##_F64                                           \
      : Sequence<COMPARE_##op##_F64,                                  \
                 I<OPCODE_COMPARE_##op, I8Op, F64Op, F64Op>> {        \
    static void Emit(A64Emitter& e, const EmitArgType& i) {           \
      if (i.src1.is_constant) {                                       \
        e.LoadConstantV(Q0, i.src1.constant());                       \
        e.FCMP(D0, i.src2);                                           \
      } else if (i.src2.is_constant) {                                \
        e.LoadConstantV(Q0, i.src2.constant());                       \
        e.FCMP(i.src1, D0);                                           \
      } else {                                                        \
        e.FCMP(i.src1, i.src2);                                       \
      }                                                               \
      e.CSET(i.dest, cond);                                           \
    }                                                                 \
  };                                                                  \
  EMITTER_OPCODE_TABLE(OPCODE_COMPARE_##op##_FLT, COMPARE_##op##_F32, \
                       COMPARE_##op##_F64);
EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(SLT, Cond::LT);  // setb
EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(SLE, Cond::LE);  // setbe
EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(SGT, Cond::GT);  // seta
EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(SGE, Cond::GE);  // setae
EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(ULT, Cond::LO);  // setb
EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(ULE, Cond::LS);  // setbe
EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(UGT, Cond::HI);  // seta
EMITTER_ASSOCIATIVE_COMPARE_FLT_XX(UGE, Cond::HS);  // setae

// ============================================================================
// OPCODE_DID_SATURATE
// ============================================================================
struct DID_SATURATE
    : Sequence<DID_SATURATE, I<OPCODE_DID_SATURATE, I8Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // Bit 27 in the FPSR is the QC bit
    e.MRS(X0, SystemReg::FPSR);
    e.UBFX(i.dest, W0, 27, 1);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_DID_SATURATE, DID_SATURATE);

// ============================================================================
// OPCODE_ADD
// ============================================================================
// TODO(benvanik): put dest/src1|2 together.
template <typename SEQ, typename REG, typename ARGS>
void EmitAddXX(A64Emitter& e, const ARGS& i) {
  SEQ::EmitCommutativeBinaryOp(
      e, i,
      [](A64Emitter& e, REG dest_src, REG src) {
        e.ADD(dest_src, dest_src, src);
      },
      [](A64Emitter& e, REG dest_src, int32_t constant) {
        e.MOV(REG(1), constant);
        e.ADD(dest_src, dest_src, REG(1));
      });
}
struct ADD_I8 : Sequence<ADD_I8, I<OPCODE_ADD, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAddXX<ADD_I8, WReg>(e, i);
  }
};
struct ADD_I16 : Sequence<ADD_I16, I<OPCODE_ADD, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAddXX<ADD_I16, WReg>(e, i);
  }
};
struct ADD_I32 : Sequence<ADD_I32, I<OPCODE_ADD, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAddXX<ADD_I32, WReg>(e, i);
  }
};
struct ADD_I64 : Sequence<ADD_I64, I<OPCODE_ADD, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAddXX<ADD_I64, XReg>(e, i);
  }
};
struct ADD_F32 : Sequence<ADD_F32, I<OPCODE_ADD, F32Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp<SReg>(
        e, i, [](A64Emitter& e, SReg dest, SReg src1, SReg src2) {
          e.FADD(dest, src1, src2);
        });
  }
};
struct ADD_F64 : Sequence<ADD_F64, I<OPCODE_ADD, F64Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    DReg src1 = i.src1.is_constant ? D1 : i.src1.reg();
    DReg src2 = i.src2.is_constant ? D2 : i.src2.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    }
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    }

    oaknut::Label do_op;
    oaknut::Label done;
    EmitPropagateInputNanF64Binary(e, i.dest.reg(), src1, src2, do_op, done);
    e.l(do_op);
    e.FADD(i.dest.reg(), src1, src2);
    e.l(done);
  }
};
struct ADD_V128 : Sequence<ADD_V128, I<OPCODE_ADD, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitWithVmxFpcr(e, [&] {
      EmitCommutativeBinaryVOp(
          e, i, [](A64Emitter& e, QReg dest, QReg src1, QReg src2) {
            e.FADD(dest.S4(), src1.S4(), src2.S4());
          });
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ADD, ADD_I8, ADD_I16, ADD_I32, ADD_I64, ADD_F32,
                     ADD_F64, ADD_V128);

// ============================================================================
// OPCODE_ADD_CARRY
// ============================================================================
// TODO(benvanik): put dest/src1|2 together.
template <typename SEQ, typename REG, typename ARGS>
void EmitAddCarryXX(A64Emitter& e, const ARGS& i) {
  // TODO(benvanik): faster setting? we could probably do some fun math tricks
  // here to get the carry flag set.
  if (i.src3.is_constant) {
    e.MOV(W0, WZR);
    if (i.src3.constant()) {
      // Set carry
      // This is implicitly "SUBS 0 - 0"
      e.CMP(W0, 0);
    } else {
      // Clear carry
      e.CMN(W0, 0);
    }
  } else {
    // If src3 is non-zero, set the carry flag
    e.CMP(i.src3.reg().toW(), 0);
    e.CSET(X0, Cond::NE);

    e.MRS(X1, SystemReg::NZCV);
    // Assign carry bit
    e.BFI(X1, X0, 29, 1);
    e.MSR(SystemReg::NZCV, X1);
  }
  SEQ::EmitCommutativeBinaryOp(
      e, i,
      [](A64Emitter& e, const REG& dest_src, const REG& src) {
        e.ADC(dest_src, dest_src, src);
      },
      [](A64Emitter& e, const REG& dest_src, int32_t constant) {
        e.MOV(REG(1), constant);
        e.ADC(dest_src, dest_src, REG(1));
      });
}
struct ADD_CARRY_I8
    : Sequence<ADD_CARRY_I8, I<OPCODE_ADD_CARRY, I8Op, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAddCarryXX<ADD_CARRY_I8, WReg>(e, i);
  }
};
struct ADD_CARRY_I16
    : Sequence<ADD_CARRY_I16, I<OPCODE_ADD_CARRY, I16Op, I16Op, I16Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAddCarryXX<ADD_CARRY_I16, WReg>(e, i);
  }
};
struct ADD_CARRY_I32
    : Sequence<ADD_CARRY_I32, I<OPCODE_ADD_CARRY, I32Op, I32Op, I32Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAddCarryXX<ADD_CARRY_I32, WReg>(e, i);
  }
};
struct ADD_CARRY_I64
    : Sequence<ADD_CARRY_I64, I<OPCODE_ADD_CARRY, I64Op, I64Op, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAddCarryXX<ADD_CARRY_I64, XReg>(e, i);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ADD_CARRY, ADD_CARRY_I8, ADD_CARRY_I16,
                     ADD_CARRY_I32, ADD_CARRY_I64);

// ============================================================================
// OPCODE_SUB
// ============================================================================
// TODO(benvanik): put dest/src1|2 together.
template <typename SEQ, typename REG, typename ARGS>
void EmitSubXX(A64Emitter& e, const ARGS& i) {
  SEQ::EmitAssociativeBinaryOp(
      e, i,
      [](A64Emitter& e, REG dest_src, REG src) {
        e.SUB(dest_src, dest_src, src);
      },
      [](A64Emitter& e, REG dest_src, int32_t constant) {
        e.MOV(REG(1), constant);
        e.SUB(dest_src, dest_src, REG(1));
      });
}
struct SUB_I8 : Sequence<SUB_I8, I<OPCODE_SUB, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitSubXX<SUB_I8, WReg>(e, i);
  }
};
struct SUB_I16 : Sequence<SUB_I16, I<OPCODE_SUB, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitSubXX<SUB_I16, WReg>(e, i);
  }
};
struct SUB_I32 : Sequence<SUB_I32, I<OPCODE_SUB, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitSubXX<SUB_I32, WReg>(e, i);
  }
};
struct SUB_I64 : Sequence<SUB_I64, I<OPCODE_SUB, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitSubXX<SUB_I64, XReg>(e, i);
  }
};
struct SUB_F32 : Sequence<SUB_F32, I<OPCODE_SUB, F32Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    EmitAssociativeBinaryVOp<SReg>(
        e, i, [](A64Emitter& e, SReg dest, SReg src1, SReg src2) {
          e.FSUB(dest, src1, src2);
        });
  }
};
struct SUB_F64 : Sequence<SUB_F64, I<OPCODE_SUB, F64Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    DReg src1 = i.src1.is_constant ? D1 : i.src1.reg();
    DReg src2 = i.src2.is_constant ? D2 : i.src2.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    }
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    }

    oaknut::Label do_op;
    oaknut::Label done;
    EmitPropagateInputNanF64Binary(e, i.dest.reg(), src1, src2, do_op, done);
    e.l(do_op);
    e.FSUB(i.dest.reg(), src1, src2);
    e.l(done);
  }
};
struct SUB_V128 : Sequence<SUB_V128, I<OPCODE_SUB, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    EmitWithVmxFpcr(e, [&] {
      EmitAssociativeBinaryVOp(
          e, i, [](A64Emitter& e, QReg dest, QReg src1, QReg src2) {
            e.FSUB(dest.S4(), src1.S4(), src2.S4());
          });
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SUB, SUB_I8, SUB_I16, SUB_I32, SUB_I64, SUB_F32,
                     SUB_F64, SUB_V128);

// ============================================================================
// OPCODE_MUL
// ============================================================================
// Sign doesn't matter here, as we don't use the high bits.
// We exploit mulx here to avoid creating too much register pressure.
struct MUL_I8 : Sequence<MUL_I8, I<OPCODE_MUL, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      assert_true(!i.src2.is_constant);
      e.MOV(W0, i.src1.constant());
      e.MUL(i.dest, W0, i.src2);
    } else if (i.src2.is_constant) {
      assert_true(!i.src1.is_constant);
      e.MOV(W0, i.src2.constant());
      e.MUL(i.dest, i.src1, W0);
    } else {
      e.MUL(i.dest, i.src1, i.src2);
    }
  }
};
struct MUL_I16 : Sequence<MUL_I16, I<OPCODE_MUL, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      assert_true(!i.src2.is_constant);
      e.MOV(W0, i.src1.constant());
      e.MUL(i.dest, W0, i.src2);
    } else if (i.src2.is_constant) {
      assert_true(!i.src1.is_constant);
      e.MOV(W0, i.src2.constant());
      e.MUL(i.dest, i.src1, W0);
    } else {
      e.MUL(i.dest, i.src1, i.src2);
    }
  }
};
struct MUL_I32 : Sequence<MUL_I32, I<OPCODE_MUL, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      assert_true(!i.src2.is_constant);
      e.MOV(W0, i.src1.constant());
      e.MUL(i.dest, W0, i.src2);
    } else if (i.src2.is_constant) {
      assert_true(!i.src1.is_constant);
      e.MOV(W0, i.src2.constant());
      e.MUL(i.dest, i.src1, W0);
    } else {
      e.MUL(i.dest, i.src1, i.src2);
    }
  }
};
struct MUL_I64 : Sequence<MUL_I64, I<OPCODE_MUL, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      assert_true(!i.src2.is_constant);
      e.MOV(X0, i.src1.constant());
      e.MUL(i.dest, X0, i.src2);
    } else if (i.src2.is_constant) {
      assert_true(!i.src1.is_constant);
      e.MOV(X0, i.src2.constant());
      e.MUL(i.dest, i.src1, X0);
    } else {
      e.MUL(i.dest, i.src1, i.src2);
    }
  }
};
struct MUL_F32 : Sequence<MUL_F32, I<OPCODE_MUL, F32Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    EmitCommutativeBinaryVOp<SReg>(
        e, i, [](A64Emitter& e, SReg dest, SReg src1, SReg src2) {
          e.FMUL(dest, src1, src2);
        });
  }
};
struct MUL_F64 : Sequence<MUL_F64, I<OPCODE_MUL, F64Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    DReg src1 = i.src1.is_constant ? D1 : i.src1.reg();
    DReg src2 = i.src2.is_constant ? D2 : i.src2.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    }
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    }

    oaknut::Label do_op;
    oaknut::Label done;
    EmitPropagateInputNanF64Binary(e, i.dest.reg(), src1, src2, do_op, done);
    e.l(do_op);
    e.FMUL(i.dest.reg(), src1, src2);
    e.l(done);
  }
};
struct MUL_V128 : Sequence<MUL_V128, I<OPCODE_MUL, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    EmitWithVmxFpcr(e, [&] {
      EmitCommutativeBinaryVOp(
          e, i, [](A64Emitter& e, QReg dest, QReg src1, QReg src2) {
            e.FMUL(dest.S4(), src1.S4(), src2.S4());
          });
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MUL, MUL_I8, MUL_I16, MUL_I32, MUL_I64, MUL_F32,
                     MUL_F64, MUL_V128);

// ============================================================================
// OPCODE_MUL_HI
// ============================================================================
struct MUL_HI_I8 : Sequence<MUL_HI_I8, I<OPCODE_MUL_HI, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & ARITHMETIC_UNSIGNED) {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.MUL(i.dest, W0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.MUL(i.dest, i.src1, W0);
      } else {
        e.MUL(i.dest, i.src1, i.src2);
      }
      e.UBFX(i.dest, i.dest, 8, 8);
    } else {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.MUL(i.dest, W0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.MUL(i.dest, i.src1, W0);
      } else {
        e.MUL(i.dest, i.src1, i.src2);
      }
      e.SBFX(i.dest, i.dest, 8, 8);
    }
  }
};
struct MUL_HI_I16
    : Sequence<MUL_HI_I16, I<OPCODE_MUL_HI, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & ARITHMETIC_UNSIGNED) {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.MUL(i.dest, W0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.MUL(i.dest, i.src1, W0);
      } else {
        e.MUL(i.dest, i.src1, i.src2);
      }
      e.UBFX(i.dest, i.dest, 16, 16);
    } else {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.MUL(i.dest, W0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.MUL(i.dest, i.src1, W0);
      } else {
        e.MUL(i.dest, i.src1, i.src2);
      }
      e.SBFX(i.dest, i.dest, 16, 16);
    }
  }
};
struct MUL_HI_I32
    : Sequence<MUL_HI_I32, I<OPCODE_MUL_HI, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & ARITHMETIC_UNSIGNED) {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.UMULL(X0, W0, i.src2);
        e.UBFX(X0, X0, 32, 32);
        e.MOV(i.dest, X0.toW());
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.UMULL(X0, W0, i.src1);
        e.UBFX(X0, X0, 32, 32);
        e.MOV(i.dest, X0.toW());
      } else {
        e.UMULL(X0, i.src1, i.src2);
        e.UBFX(X0, X0, 32, 32);
        e.MOV(i.dest, X0.toW());
      }
    } else {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.SMULL(X0, W0, i.src2);
        e.SBFX(X0, X0, 32, 32);
        e.MOV(i.dest, X0.toW());
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.SMULL(X0, W0, i.src1);
        e.SBFX(X0, X0, 32, 32);
        e.MOV(i.dest, X0.toW());
      } else {
        e.SMULL(X0, i.src1, i.src2);
        e.SBFX(X0, X0, 32, 32);
        e.MOV(i.dest, X0.toW());
      }
    }
  }
};
struct MUL_HI_I64
    : Sequence<MUL_HI_I64, I<OPCODE_MUL_HI, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & ARITHMETIC_UNSIGNED) {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(X0, i.src1.constant());
        e.UMULH(i.dest, X0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(X0, i.src2.constant());
        e.UMULH(i.dest, i.src1, X0);
      } else {
        e.UMULH(i.dest, i.src1, i.src2);
      }
    } else {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(X0, i.src1.constant());
        e.SMULH(i.dest, X0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(X0, i.src2.constant());
        e.SMULH(i.dest, i.src1, X0);
      } else {
        e.SMULH(i.dest, i.src1, i.src2);
      }
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MUL_HI, MUL_HI_I8, MUL_HI_I16, MUL_HI_I32,
                     MUL_HI_I64);

// ============================================================================
// OPCODE_DIV
// ============================================================================
struct DIV_I8 : Sequence<DIV_I8, I<OPCODE_DIV, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & ARITHMETIC_UNSIGNED) {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.UDIV(i.dest, W0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.UDIV(i.dest, i.src1, W0);
      } else {
        e.UDIV(i.dest, i.src1, i.src2);
      }
      e.UXTB(i.dest, i.dest);
    } else {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.SDIV(i.dest, W0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.SDIV(i.dest, i.src1, W0);
      } else {
        e.SDIV(i.dest, i.src1, i.src2);
      }
      e.SXTB(i.dest, i.dest);
    }
  }
};
struct DIV_I16 : Sequence<DIV_I16, I<OPCODE_DIV, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & ARITHMETIC_UNSIGNED) {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.UDIV(i.dest, W0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.UDIV(i.dest, i.src1, W0);
      } else {
        e.UDIV(i.dest, i.src1, i.src2);
      }
      e.UXTH(i.dest, i.dest);
    } else {
      if (i.src1.is_constant) {
        assert_true(!i.src2.is_constant);
        e.MOV(W0, i.src1.constant());
        e.SDIV(i.dest, W0, i.src2);
      } else if (i.src2.is_constant) {
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.SDIV(i.dest, i.src1, W0);
      } else {
        e.SDIV(i.dest, i.src1, i.src2);
      }
      e.SXTH(i.dest, i.dest);
    }
  }
};
struct DIV_I32 : Sequence<DIV_I32, I<OPCODE_DIV, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & ARITHMETIC_UNSIGNED) {
      if (i.src2.is_constant) {
        if (i.src2.constant() == 0) {
          e.MOV(i.dest, 0);
          return;
        }
        assert_true(!i.src1.is_constant);
        e.MOV(W0, i.src2.constant());
        e.UDIV(i.dest, i.src1, W0);
      } else {
        oaknut::Label div_zero, done;
        e.CBZ(i.src2, div_zero);
        if (i.src1.is_constant) {
          e.MOV(W0, i.src1.constant());
          e.UDIV(i.dest, W0, i.src2);
        } else {
          e.UDIV(i.dest, i.src1, i.src2);
        }
        e.B(done);
        e.l(div_zero);
        e.MOV(i.dest, 0);
        e.l(done);
      }
    } else {
      if (i.src2.is_constant) {
        const int32_t divisor = i.src2.constant();
        if (divisor == 0) {
          e.MOV(i.dest, 0);
          return;
        }
        assert_true(!i.src1.is_constant);
        if (divisor == -1) {
          oaknut::Label do_div, done;
          e.MOV(W1, 0x80000000u);
          e.CMP(i.src1, W1);
          e.B(Cond::NE, do_div);
          e.MOV(i.dest, 0);
          e.B(done);
          e.l(do_div);
          e.MOV(W0, divisor);
          e.SDIV(i.dest, i.src1, W0);
          e.l(done);
          return;
        }
        e.MOV(W0, divisor);
        e.SDIV(i.dest, i.src1, W0);
      } else {
        oaknut::Label div_zero, do_div, done;
        e.CBZ(i.src2, div_zero);
        e.MOV(W2, static_cast<uint32_t>(-1));
        e.CMP(i.src2, W2);
        e.B(Cond::NE, do_div);
        e.MOV(W1, 0x80000000u);
        if (i.src1.is_constant) {
          e.MOV(W0, i.src1.constant());
          e.CMP(W0, W1);
        } else {
          e.CMP(i.src1, W1);
        }
        e.B(Cond::NE, do_div);
        e.MOV(i.dest, 0);
        e.B(done);
        e.l(do_div);
        if (i.src1.is_constant) {
          e.MOV(W0, i.src1.constant());
          e.SDIV(i.dest, W0, i.src2);
        } else {
          e.SDIV(i.dest, i.src1, i.src2);
        }
        e.B(done);
        e.l(div_zero);
        e.MOV(i.dest, 0);
        e.l(done);
      }
    }
  }
};
struct DIV_I64 : Sequence<DIV_I64, I<OPCODE_DIV, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & ARITHMETIC_UNSIGNED) {
      if (i.src2.is_constant) {
        if (i.src2.constant() == 0) {
          e.MOV(i.dest, 0);
          return;
        }
        assert_true(!i.src1.is_constant);
        e.MOV(X0, i.src2.constant());
        e.UDIV(i.dest, i.src1, X0);
      } else {
        oaknut::Label div_zero, done;
        e.CBZ(i.src2, div_zero);
        if (i.src1.is_constant) {
          e.MOV(X0, i.src1.constant());
          e.UDIV(i.dest, X0, i.src2);
        } else {
          e.UDIV(i.dest, i.src1, i.src2);
        }
        e.B(done);
        e.l(div_zero);
        e.MOV(i.dest, 0);
        e.l(done);
      }
    } else {
      if (i.src2.is_constant) {
        const int64_t divisor = i.src2.constant();
        if (divisor == 0) {
          e.MOV(i.dest, 0);
          return;
        }
        assert_true(!i.src1.is_constant);
        if (divisor == -1) {
          oaknut::Label do_div, done;
          e.MOV(X1, 0x8000000000000000ull);
          e.CMP(i.src1, X1);
          e.B(Cond::NE, do_div);
          e.MOV(i.dest, 0);
          e.B(done);
          e.l(do_div);
          e.MOV(X0, divisor);
          e.SDIV(i.dest, i.src1, X0);
          e.l(done);
          return;
        }
        e.MOV(X0, divisor);
        e.SDIV(i.dest, i.src1, X0);
      } else {
        oaknut::Label div_zero, do_div, done;
        e.CBZ(i.src2, div_zero);
        e.MOV(X2, static_cast<uint64_t>(-1));
        e.CMP(i.src2, X2);
        e.B(Cond::NE, do_div);
        e.MOV(X1, 0x8000000000000000ull);
        if (i.src1.is_constant) {
          e.MOV(X0, i.src1.constant());
          e.CMP(X0, X1);
        } else {
          e.CMP(i.src1, X1);
        }
        e.B(Cond::NE, do_div);
        e.MOV(i.dest, 0);
        e.B(done);
        e.l(do_div);
        if (i.src1.is_constant) {
          e.MOV(X0, i.src1.constant());
          e.SDIV(i.dest, X0, i.src2);
        } else {
          e.SDIV(i.dest, i.src1, i.src2);
        }
        e.B(done);
        e.l(div_zero);
        e.MOV(i.dest, 0);
        e.l(done);
      }
    }
  }
};
struct DIV_F32 : Sequence<DIV_F32, I<OPCODE_DIV, F32Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    EmitAssociativeBinaryVOp<SReg>(
        e, i, [](A64Emitter& e, SReg dest, SReg src1, SReg src2) {
          e.FDIV(dest, src1, src2);
        });
  }
};
struct DIV_F64 : Sequence<DIV_F64, I<OPCODE_DIV, F64Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    DReg src1 = i.src1.is_constant ? D1 : i.src1.reg();
    DReg src2 = i.src2.is_constant ? D2 : i.src2.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    }
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    }

    oaknut::Label do_op;
    oaknut::Label done;
    EmitPropagateInputNanF64Binary(e, i.dest.reg(), src1, src2, do_op, done);
    e.l(do_op);
    e.FDIV(i.dest.reg(), src1, src2);
    e.l(done);
  }
};
struct DIV_V128 : Sequence<DIV_V128, I<OPCODE_DIV, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    EmitAssociativeBinaryVOp(
        e, i, [](A64Emitter& e, QReg dest, QReg src1, QReg src2) {
          e.FDIV(dest.S4(), src1.S4(), src2.S4());
        });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_DIV, DIV_I8, DIV_I16, DIV_I32, DIV_I64, DIV_F32,
                     DIV_F64, DIV_V128);

// ============================================================================
// OPCODE_MUL_ADD
// ============================================================================
// d = 1 * 2 + 3
// $0 = $1x$0 + $2
struct MUL_ADD_F32
    : Sequence<MUL_ADD_F32, I<OPCODE_MUL_ADD, F32Op, F32Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    SReg src3 = S3;
    if (i.src3.is_constant) {
      e.LoadConstantV(src3.toQ(), i.src3.constant());
    } else {
      src3 = i.src3.reg();
    }

    SReg src2 = S2;
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    } else {
      src2 = i.src2.reg();
    }

    SReg src1 = S1;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    } else {
      src1 = i.src1.reg();
    }

    e.FMADD(i.dest, src1, src2, src3);
  }
};
struct MUL_ADD_F64
    : Sequence<MUL_ADD_F64, I<OPCODE_MUL_ADD, F64Op, F64Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    DReg src1 = D1;
    DReg src2 = D2;
    DReg src3 = D3;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    } else {
      e.FMOV(src1, i.src1.reg());
    }
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    } else {
      e.FMOV(src2, i.src2.reg());
    }
    if (i.src3.is_constant) {
      e.LoadConstantV(src3.toQ(), i.src3.constant());
    } else {
      e.FMOV(src3, i.src3.reg());
    }

    oaknut::Label do_op;
    oaknut::Label done;
    EmitPropagateInputNanF64Ternary(e, i.dest.reg(), src1, src2, src3, do_op,
                                    done);
    e.l(do_op);
    e.FMADD(i.dest.reg(), src1, src2, src3);
    e.l(done);
  }
};
struct MUL_ADD_V128
    : Sequence<MUL_ADD_V128,
               I<OPCODE_MUL_ADD, V128Op, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitWithVmxFpcr(e, [&] {
      const QReg dest = i.dest.reg();
      const QReg temp1 = Q0;
      const QReg temp2 = Q1;
      const QReg temp3 = Q2;

      if (i.src1.is_constant) {
        e.LoadConstantV(temp1, i.src1.constant());
      } else {
        e.MOV(temp1.B16(), i.src1.reg().B16());
      }

      if (i.src2.is_constant) {
        e.LoadConstantV(temp2, i.src2.constant());
      } else {
        e.MOV(temp2.B16(), i.src2.reg().B16());
      }

      if (i.src3.is_constant) {
        e.LoadConstantV(temp3, i.src3.constant());
      } else {
        e.MOV(temp3.B16(), i.src3.reg().B16());
      }

      e.MOV(dest.B16(), temp3.B16());
      e.FMLA(dest.S4(), temp1.S4(), temp2.S4());

      EmitSelectFirstQNanLanesF32x4(e, dest, temp1, temp2, temp3);
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MUL_ADD, MUL_ADD_F32, MUL_ADD_F64, MUL_ADD_V128);

// ============================================================================
// OPCODE_MUL_SUB
// ============================================================================
// d = 1 * 2 - 3
// $0 = $2x$0 - $3
// TODO(benvanik): use other forms (132/213/etc) to avoid register shuffling.
// dest could be src2 or src3 - need to ensure it's not before overwriting dest
// perhaps use other 132/213/etc
// Forms:
// - 132 -> $1 = $1 * $3 - $2
// - 213 -> $1 = $2 * $1 - $3
// - 231 -> $1 = $2 * $3 - $1
struct MUL_SUB_F32
    : Sequence<MUL_SUB_F32, I<OPCODE_MUL_SUB, F32Op, F32Op, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    SReg src3(1);
    if (i.src3.is_constant) {
      src3 = S1;
      e.LoadConstantV(src3.toQ(), i.src3.constant());
    } else {
      // If i.dest == i.src3, back up i.src3 so we don't overwrite it.
      src3 = i.src3.reg();
      if (i.dest.reg().index() == i.src3.reg().index()) {
        e.FMOV(S1, i.src3);
        src3 = S1;
      }
    }

    // Multiply operation is commutative.
    EmitCommutativeBinaryVOp<SReg>(
        e, i, [](A64Emitter& e, SReg dest, SReg src1, SReg src2) {
          e.FMUL(dest, src1, src2);  // $0 = $1 * $2
        });

    e.FSUB(i.dest, i.dest, src3);  // $0 = $1 - $2
  }
};
struct MUL_SUB_F64
    : Sequence<MUL_SUB_F64, I<OPCODE_MUL_SUB, F64Op, F64Op, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    DReg src1 = D1;
    DReg src2 = D2;
    DReg src3 = D3;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    } else {
      e.FMOV(src1, i.src1.reg());
    }
    if (i.src2.is_constant) {
      e.LoadConstantV(src2.toQ(), i.src2.constant());
    } else {
      e.FMOV(src2, i.src2.reg());
    }
    if (i.src3.is_constant) {
      e.LoadConstantV(src3.toQ(), i.src3.constant());
    } else {
      e.FMOV(src3, i.src3.reg());
    }

    oaknut::Label do_op;
    oaknut::Label done;
    EmitPropagateInputNanF64Ternary(e, i.dest.reg(), src1, src2, src3, do_op,
                                    done);
    e.l(do_op);
    e.FMUL(i.dest.reg(), src1, src2);
    e.FSUB(i.dest.reg(), i.dest.reg(), src3);
    e.l(done);
  }
};
struct MUL_SUB_V128
    : Sequence<MUL_SUB_V128,
               I<OPCODE_MUL_SUB, V128Op, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitWithVmxFpcr(e, [&] {
      const QReg dest = i.dest.reg();
      QReg temp1 = Q0;
      QReg temp2 = Q1;
      QReg temp3 = Q2;

      if (i.src1.is_constant) {
        e.LoadConstantV(temp1, i.src1.constant());
      } else {
        e.MOV(temp1.B16(), i.src1.reg().B16());
      }

      if (i.src2.is_constant) {
        e.LoadConstantV(temp2, i.src2.constant());
      } else {
        e.MOV(temp2.B16(), i.src2.reg().B16());
      }

      if (i.src3.is_constant) {
        e.LoadConstantV(temp3, i.src3.constant());
      } else {
        e.MOV(temp3.B16(), i.src3.reg().B16());
      }

      // a * b - c == a * b + (-c), keeping one final rounding.
      e.FNEG(dest.S4(), temp3.S4());
      e.FMLA(dest.S4(), temp1.S4(), temp2.S4());
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MUL_SUB, MUL_SUB_F32, MUL_SUB_F64, MUL_SUB_V128);

// ============================================================================
// OPCODE_NEG
// ============================================================================
// TODO(benvanik): put dest/src1 together.
template <typename SEQ, typename REG, typename ARGS>
void EmitNegXX(A64Emitter& e, const ARGS& i) {
  SEQ::EmitUnaryOp(
      e, i, [](A64Emitter& e, REG dest_src) { e.NEG(dest_src, dest_src); });
}
struct NEG_I8 : Sequence<NEG_I8, I<OPCODE_NEG, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitNegXX<NEG_I8, WReg>(e, i);
  }
};
struct NEG_I16 : Sequence<NEG_I16, I<OPCODE_NEG, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitNegXX<NEG_I16, WReg>(e, i);
  }
};
struct NEG_I32 : Sequence<NEG_I32, I<OPCODE_NEG, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitNegXX<NEG_I32, WReg>(e, i);
  }
};
struct NEG_I64 : Sequence<NEG_I64, I<OPCODE_NEG, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitNegXX<NEG_I64, XReg>(e, i);
  }
};
struct NEG_F32 : Sequence<NEG_F32, I<OPCODE_NEG, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FNEG(i.dest, i.src1);
  }
};
struct NEG_F64 : Sequence<NEG_F64, I<OPCODE_NEG, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // PPC fnmadd/fnmsub are lowered to NEG(MUL_ADD/MUL_SUB).
    // For NaN results, PPC preserves sign/payload instead of negating them.
    if (!i.src1.is_constant && IsMulAddOrMulSubDef(i.src1.value)) {
      DReg src_for_nan = i.src1.reg();
      if (i.dest.reg().index() == i.src1.reg().index()) {
        src_for_nan = i.src1.reg().index() == 0 ? D1 : D0;
        e.FMOV(src_for_nan, i.src1.reg());
      }

      oaknut::Label done;
      e.FNEG(i.dest, i.src1);
      e.FCMP(src_for_nan, src_for_nan);
      e.B(Cond::VC, done);
      e.FMOV(i.dest.reg(), src_for_nan);
      e.l(done);
      return;
    }

    e.FNEG(i.dest, i.src1);
  }
};
struct NEG_V128 : Sequence<NEG_V128, I<OPCODE_NEG, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(!i.instr->flags);
    QReg src = i.src1.is_constant ? Q0 : i.src1.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    e.FNEG(i.dest.reg().S4(), src.S4());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_NEG, NEG_I8, NEG_I16, NEG_I32, NEG_I64, NEG_F32,
                     NEG_F64, NEG_V128);

// ============================================================================
// OPCODE_ABS
// ============================================================================
struct ABS_F32 : Sequence<ABS_F32, I<OPCODE_ABS, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FABS(i.dest, i.src1);
  }
};
struct ABS_F64 : Sequence<ABS_F64, I<OPCODE_ABS, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.FABS(i.dest, i.src1);
  }
};
struct ABS_V128 : Sequence<ABS_V128, I<OPCODE_ABS, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    QReg src = i.src1.is_constant ? Q0 : i.src1.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    e.FABS(i.dest.reg().S4(), src.S4());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ABS, ABS_F32, ABS_F64, ABS_V128);

// ============================================================================
// OPCODE_SQRT
// ============================================================================
struct SQRT_F32 : Sequence<SQRT_F32, I<OPCODE_SQRT, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    SReg src1 = S0;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    } else {
      src1 = i.src1.reg();
    }
    e.FSQRT(i.dest, src1);
  }
};
struct SQRT_F64 : Sequence<SQRT_F64, I<OPCODE_SQRT, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    DReg src1 = D0;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    } else {
      src1 = i.src1.reg();
    }
    e.FSQRT(i.dest, src1);
  }
};
struct SQRT_V128 : Sequence<SQRT_V128, I<OPCODE_SQRT, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitWithVmxFpcr(e, [&] {
      if (i.src1.is_constant) {
        e.LoadConstantV(Q0, i.src1.constant());
        e.FSQRT(i.dest.reg().S4(), Q0.S4());
      } else {
        e.FSQRT(i.dest.reg().S4(), i.src1.reg().S4());
      }
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SQRT, SQRT_F32, SQRT_F64, SQRT_V128);

// ============================================================================
// OPCODE_RSQRT
// ============================================================================
inline uint32_t CountLeadingZeros32(uint32_t value) {
  return value ? static_cast<uint32_t>(__builtin_clz(value)) : 32u;
}

inline uint32_t CountLeadingZeros64(uint64_t value) {
  return value ? static_cast<uint32_t>(__builtin_clzll(value)) : 64u;
}

inline uint32_t FloatToBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float BitsToFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline uint64_t DoubleToBits(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline double BitsToDouble(uint64_t bits) {
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Copied from x64 backend to match PPC estimate behavior.
static constexpr uint32_t kVrsqrteTable[32] = {
    0x0568B4FDu, 0x04F3AF97u, 0x048DAAA5u, 0x0435A618u, 0x03E7A1E4u,
    0x03A29DFEu, 0x03659A5Cu, 0x032E96F8u, 0x02FC93CAu, 0x02D090CEu,
    0x02A88DFEu, 0x02838B57u, 0x026188D4u, 0x02438673u, 0x02268431u,
    0x020B820Bu, 0x03D27FFAu, 0x03807C29u, 0x033878AAu, 0x02F97572u,
    0x02C27279u, 0x02926FB7u, 0x02666D26u, 0x023F6AC0u, 0x021D6881u,
    0x01FD6665u, 0x01E16468u, 0x01C76287u, 0x01AF60C1u, 0x01995F12u,
    0x01855D79u, 0x01735BF4u,
};

static constexpr uint8_t kFrsqrteTable[16] = {
    241u, 216u, 192u, 168u, 152u, 136u, 128u, 112u,
    96u,  76u,  60u,  48u,  32u,  24u,  16u,  8u,
};

inline uint32_t EmulateVrsqrteScalarBits(uint32_t src_bits,
                                         bool njm_enabled = true) {
  uint32_t r8d = src_bits;
  float input = BitsToFloat(r8d);
  uint32_t ecx = r8d & 0x7FFFFFu;
  uint32_t edx = ecx;
  int32_t eax = 0;
  uint32_t r9d = 0;

  if (r8d == 0xFF800000u) {
    return 0x7FC00000u;
  }

  if ((r8d & 0x7F800000u) == 0 && ecx != 0) {
    if (!njm_enabled) {
      r9d = r8d & 0x7FFFFFFFu;
      if (r9d == 0x00400000u) {
        return ((~r8d >> 31) != 0) ? 0x5F34FD00u : 0x7FC00000u;
      }
      ecx = CountLeadingZeros32(ecx);
      r9d = 9;
      eax = -118;
      edx = static_cast<uint32_t>(static_cast<int32_t>(ecx) - 8);
      r9d -= ecx;
      eax -= static_cast<int32_t>(ecx);
      edx = r8d << edx;
      edx &= 0x7FFFFEu;
      goto compute_from_fields;
    }
    return ((~r8d >> 31) != 0) ? 0x7F800000u : 0xFF800000u;
  }

  r9d = (r8d >> 23) & 0xFFu;
  eax = static_cast<int32_t>(r9d) - 127;
  if (r9d == 255u) {
    if (!(input < 0.0f || std::isnan(input))) {
      if (ecx == 0) {
        return 0u;
      }
    }
    return r8d | 0x00400000u;
  }

compute_from_fields:
  if (eax == 128) {
    if (0.0f > input) {
      if (edx == 0) {
        return 0x7FC00000u;
      }
      return r8d | 0x00400000u;
    }
  }

  if (edx == 0) {
    if (r9d == 0u) {
      return ((~r8d >> 31) != 0) ? 0x7F800000u : 0xFF800000u;
    }
  } else if (eax == 128) {
    return r8d | 0x00400000u;
  }

  if (0.0f > input) {
    return 0x7FC00000u;
  }

  ecx = 127;
  eax <<= 4;
  ecx -= static_cast<int32_t>(r9d);
  r9d = edx;
  eax &= 16;
  edx >>= 9;
  r9d >>= 19;
  edx &= 1023;
  ecx >>= 1;
  eax |= static_cast<int32_t>(r9d);
  eax ^= 16;

  uint32_t table_entry = kVrsqrteTable[static_cast<uint32_t>(eax) & 31];
  eax = static_cast<int32_t>(table_entry);
  r9d = table_entry >> 16;
  edx *= r9d;
  eax <<= 10;
  eax &= 0x3FFFC00;
  eax -= static_cast<int32_t>(edx);
  if (((static_cast<uint32_t>(eax) >> 25) & 1u) == 0) {
    edx = static_cast<uint32_t>(eax) & 0x1FFFFFFu;
    ecx += 6;
    uint32_t lz = CountLeadingZeros32(edx);
    r9d = lz - 6;
    ecx -= static_cast<int32_t>(lz);
    eax = static_cast<int32_t>(static_cast<uint32_t>(eax) << r9d);
  }

  if ((eax & 0x5) != 0 && (eax & 0x2) != 0) {
    eax += 4;
  }

  ecx <<= 23;
  r8d &= 0x80000000u;
  eax = static_cast<int32_t>(static_cast<uint32_t>(eax) >> 2);
  ecx += 0x3F800000;
  eax &= 0x7FFFFF;
  uint32_t out = static_cast<uint32_t>(ecx) | r8d | static_cast<uint32_t>(eax);
  // Apply DAZ/FTZ behavior used by PPC estimate path.
  if ((out & 0x7F800000u) == 0 && (out & 0x007FFFFFu) != 0) {
    out &= 0x80000000u;
  }
  return out;
}

inline uint64_t EmulateFrsqrteBits(uint64_t src_bits,
                                   bool non_ieee_mode = false) {
  uint64_t rax = src_bits;
  constexpr uint64_t kSignMask = 0x8000000000000000ULL;
  constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;

  if (non_ieee_mode) {
    uint64_t shifted = rax << 12;
    if (shifted != 0 && (rax & kExpMask) == 0) {
      return (rax & kSignMask) | kExpMask;
    }
  }

  uint64_t rcx = rax + rax;
  if (rcx == 0) {
    return (rax & kSignMask) | kExpMask;
  }

  if (((~rax) & kExpMask) == 0) {
    if (rax == kExpMask) {
      return 0ULL;
    }
    uint64_t shifted = rax << 12;
    if (shifted == 0 && BitsToDouble(rax) < 0.0) {
      return 0x7FF8000000000000ULL;
    }
    return rax | 0x7FF8000000000000ULL;
  }

  if (0.0 > BitsToDouble(rax)) {
    return 0x7FF8000000000000ULL;
  }

  uint64_t exponent = (rax >> 52) & 0x7FFULL;
  uint64_t mantissa = rax & 0x000FFFFFFFFFFFFFULL;

  if (mantissa != 0 && exponent == 0) {
    uint32_t lz = CountLeadingZeros64(mantissa);
    mantissa <<= static_cast<uint32_t>(static_cast<int32_t>(lz) - 11);
    exponent = static_cast<uint64_t>(12 - static_cast<int32_t>(lz));
  }

  uint32_t edx = static_cast<uint32_t>(exponent * 8);
  uint32_t eax = static_cast<uint32_t>(mantissa >> 49) & 7u;
  int32_t exp_unbiased = static_cast<int32_t>(exponent) - 1023;
  edx &= 8u;
  exp_unbiased >>= 1;
  eax |= edx;
  edx = static_cast<uint32_t>(1022 - exp_unbiased);
  eax ^= 8u;
  uint64_t out = (static_cast<uint64_t>(edx) << 52) |
                 (static_cast<uint64_t>(kFrsqrteTable[eax & 15u]) << 44);
  return out;
}

inline float64x2_t EmulateRsqrtF64(void* raw_context, std::byte src[16]) {
  bool non_ieee_mode = false;
  if (raw_context) {
    auto* backend_context = reinterpret_cast<A64BackendContext*>(
        reinterpret_cast<std::byte*>(raw_context) - sizeof(A64BackendContext));
    non_ieee_mode = backend_context->non_ieee_mode != 0;
  }
  uint64_t in_bits;
  std::memcpy(&in_bits, src, sizeof(in_bits));
  uint64_t out_bits = EmulateFrsqrteBits(in_bits, non_ieee_mode);
  uint64x2_t out = vdupq_n_u64(0);
  out = vsetq_lane_u64(out_bits, out, 0);
  return vreinterpretq_f64_u64(out);
}

inline float32x4_t EmulateRsqrtV128(void* raw_context, std::byte src[16]) {
  bool njm_enabled = true;
  if (raw_context) {
    auto* backend_context = reinterpret_cast<A64BackendContext*>(
        reinterpret_cast<std::byte*>(raw_context) - sizeof(A64BackendContext));
    njm_enabled = backend_context->njm_enabled != 0;
  }
  alignas(16) uint32_t lanes[4];
  std::memcpy(lanes, src, sizeof(lanes));
  for (uint32_t& lane : lanes) {
    lane = EmulateVrsqrteScalarBits(lane, njm_enabled);
  }
  return vreinterpretq_f32_u32(vld1q_u32(lanes));
}

// Altivec guarantees an error of < 1/4096 for vrsqrtefp
struct RSQRT_F32 : Sequence<RSQRT_F32, I<OPCODE_RSQRT, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    SReg src1 = S0;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    } else {
      src1 = i.src1.reg();
    }
    e.FRSQRTE(i.dest, src1);
  }
};
struct RSQRT_F64 : Sequence<RSQRT_F64, I<OPCODE_RSQRT, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateRsqrtF64));
    e.FMOV(i.dest, D0);
  }
};
struct RSQRT_V128 : Sequence<RSQRT_V128, I<OPCODE_RSQRT, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateRsqrtV128));
    e.MOV(i.dest.reg().B16(), Q0.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_RSQRT, RSQRT_F32, RSQRT_F64, RSQRT_V128);

// ============================================================================
// OPCODE_RECIP
// ============================================================================
// Altivec guarantees an error of < 1/4096 for vrefp
struct RECIP_F32 : Sequence<RECIP_F32, I<OPCODE_RECIP, F32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    SReg src1 = S0;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    } else {
      src1 = i.src1.reg();
    }
    SReg one = src1.index() == 0 ? S1 : S0;
    e.MOV(W2, 0x3F800000u);
    e.FMOV(one, W2);
    e.FDIV(i.dest, one, src1);
  }
};
struct RECIP_F64 : Sequence<RECIP_F64, I<OPCODE_RECIP, F64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    DReg src1 = D0;
    if (i.src1.is_constant) {
      e.LoadConstantV(src1.toQ(), i.src1.constant());
    } else {
      src1 = i.src1.reg();
    }
    DReg one = src1.index() == 0 ? D1 : D0;
    e.MOV(X2, 0x3FF0000000000000ULL);
    e.FMOV(one, X2);
    e.FDIV(i.dest, one, src1);
  }
};
struct RECIP_V128 : Sequence<RECIP_V128, I<OPCODE_RECIP, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitWithVmxFpcr(e, [&] {
      QReg src = i.src1.is_constant ? Q0 : i.src1.reg();
      if (i.src1.is_constant) {
        e.LoadConstantV(src, i.src1.constant());
      }
      QReg one = src.index() == 0 ? Q1 : Q0;
      e.FMOV(one.S4(), FImm8(0, 7, 0));
      e.FDIV(i.dest.reg().S4(), one.S4(), src.S4());
    });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_RECIP, RECIP_F32, RECIP_F64, RECIP_V128);

// ============================================================================
// OPCODE_POW2
// ============================================================================
// TODO(benvanik): use approx here:
//     https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
struct POW2_F32 : Sequence<POW2_F32, I<OPCODE_POW2, F32Op, F32Op>> {
  static float32x4_t EmulatePow2(void*, std::byte src[16]) {
    float src_value;
    vst1q_lane_f32(&src_value, vld1q_u8(reinterpret_cast<const uint8_t*>(src)),
                   0);
    const float result = std::exp2(src_value);
    return vld1q_lane_f32(&result,
                          vld1q_u8(reinterpret_cast<const uint8_t*>(src)), 0);
  }
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_always();
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulatePow2));
    e.FMOV(i.dest, S0);
  }
};
struct POW2_F64 : Sequence<POW2_F64, I<OPCODE_POW2, F64Op, F64Op>> {
  static float64x2_t EmulatePow2(void*, std::byte src[16]) {
    double src_value;
    vst1q_lane_f64(&src_value, vld1q_u8(reinterpret_cast<const uint8_t*>(src)),
                   0);
    const double result = std::exp2(src_value);
    return vld1q_lane_f64(&result,
                          vld1q_u8(reinterpret_cast<const uint8_t*>(src)), 0);
  }
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_always();
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulatePow2));
    e.FMOV(i.dest, D0);
  }
};
struct POW2_V128 : Sequence<POW2_V128, I<OPCODE_POW2, V128Op, V128Op>> {
  static float32x4_t EmulatePow2(void*, std::byte src[16]) {
    alignas(16) float values[4];
    vst1q_f32(values, vld1q_u8(reinterpret_cast<const uint8_t*>(src)));
    for (size_t i = 0; i < 4; ++i) {
      values[i] = std::exp2(values[i]);
    }
    return vld1q_f32(values);
  }
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulatePow2));
    e.MOV(i.dest.reg().B16(), Q0.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_POW2, POW2_F32, POW2_F64, POW2_V128);

// ============================================================================
// OPCODE_LOG2
// ============================================================================
// TODO(benvanik): use approx here:
//     https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
// TODO(benvanik): this emulated fn destroys all xmm registers! don't do it!
struct LOG2_F32 : Sequence<LOG2_F32, I<OPCODE_LOG2, F32Op, F32Op>> {
  static float32x4_t EmulateLog2(void*, std::byte src[16]) {
    float src_value;
    vst1q_lane_f32(&src_value, vld1q_u8(reinterpret_cast<const uint8_t*>(src)),
                   0);
    float result = std::log2(src_value);
    return vld1q_lane_f32(&result,
                          vld1q_u8(reinterpret_cast<const uint8_t*>(src)), 0);
  }
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_always();
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateLog2));
    e.FMOV(i.dest, S0);
  }
};
struct LOG2_F64 : Sequence<LOG2_F64, I<OPCODE_LOG2, F64Op, F64Op>> {
  static float64x2_t EmulateLog2(void*, std::byte src[16]) {
    double src_value;
    vst1q_lane_f64(&src_value, vld1q_u8(reinterpret_cast<const uint8_t*>(src)),
                   0);
    double result = std::log2(src_value);
    return vld1q_lane_f64(&result,
                          vld1q_u8(reinterpret_cast<const uint8_t*>(src)), 0);
  }
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_always();
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateLog2));
    e.FMOV(i.dest, D0);
  }
};
struct LOG2_V128 : Sequence<LOG2_V128, I<OPCODE_LOG2, V128Op, V128Op>> {
  static float32x4_t EmulateLog2(void*, std::byte src[16]) {
    alignas(16) float values[4];
    vst1q_f32(values, vld1q_u8(reinterpret_cast<const uint8_t*>(src)));
    for (size_t i = 0; i < 4; ++i) {
      values[i] = std::log2(values[i]);
    }
    return vld1q_f32(values);
  }
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateLog2));
    e.MOV(i.dest.reg().B16(), Q0.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOG2, LOG2_F32, LOG2_F64, LOG2_V128);

// ============================================================================
// OPCODE_DOT_PRODUCT_3
// ============================================================================
inline uint32_t EmulateDotProductBits(const std::byte src1[16],
                                      const std::byte src2[16], bool dot3) {
  alignas(16) uint32_t lhs_bits[4];
  alignas(16) uint32_t rhs_bits[4];
  std::memcpy(lhs_bits, src1, sizeof(lhs_bits));
  std::memcpy(rhs_bits, src2, sizeof(rhs_bits));

  if (dot3) {
    lhs_bits[3] = 0;
    rhs_bits[3] = 0;
  }

  const double p0 = static_cast<double>(BitsToFloat(lhs_bits[0])) *
                    static_cast<double>(BitsToFloat(rhs_bits[0]));
  const double p1 = static_cast<double>(BitsToFloat(lhs_bits[1])) *
                    static_cast<double>(BitsToFloat(rhs_bits[1]));
  const double p2 = static_cast<double>(BitsToFloat(lhs_bits[2])) *
                    static_cast<double>(BitsToFloat(rhs_bits[2]));
  const double p3 = static_cast<double>(BitsToFloat(lhs_bits[3])) *
                    static_cast<double>(BitsToFloat(rhs_bits[3]));

  const double sum = dot3 ? ((p0 + p2) + p1) : ((p0 + p2) + (p1 + p3));
  if (std::isfinite(sum) &&
      std::fabs(sum) > static_cast<double>(std::numeric_limits<float>::max())) {
    return 0x7FC00000u;
  }

  float result = static_cast<float>(sum);
  uint32_t result_bits;
  std::memcpy(&result_bits, &result, sizeof(result_bits));
  return result_bits;
}

inline float32x4_t EmulateDotProduct3V128(void*, std::byte src1[16],
                                          std::byte src2[16]) {
  uint32_t result_bits = EmulateDotProductBits(src1, src2, true);
  return vreinterpretq_f32_u32(vdupq_n_u32(result_bits));
}

struct DOT_PRODUCT_3_V128
    : Sequence<DOT_PRODUCT_3_V128,
               I<OPCODE_DOT_PRODUCT_3, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    if (i.src2.is_constant) {
      e.ADD(e.GetNativeParam(1), SP, e.StashConstantV(1, i.src2.constant()));
    } else {
      e.ADD(e.GetNativeParam(1), SP, e.StashV(1, i.src2.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateDotProduct3V128));
    e.MOV(i.dest.reg().B16(), Q0.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_DOT_PRODUCT_3, DOT_PRODUCT_3_V128);

// ============================================================================
// OPCODE_DOT_PRODUCT_4
// ============================================================================
inline float32x4_t EmulateDotProduct4V128(void*, std::byte src1[16],
                                          std::byte src2[16]) {
  uint32_t result_bits = EmulateDotProductBits(src1, src2, false);
  return vreinterpretq_f32_u32(vdupq_n_u32(result_bits));
}

struct DOT_PRODUCT_4_V128
    : Sequence<DOT_PRODUCT_4_V128,
               I<OPCODE_DOT_PRODUCT_4, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.ADD(e.GetNativeParam(0), SP, e.StashConstantV(0, i.src1.constant()));
    } else {
      e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1.reg().toQ()));
    }
    if (i.src2.is_constant) {
      e.ADD(e.GetNativeParam(1), SP, e.StashConstantV(1, i.src2.constant()));
    } else {
      e.ADD(e.GetNativeParam(1), SP, e.StashV(1, i.src2.reg().toQ()));
    }
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateDotProduct4V128));
    e.MOV(i.dest.reg().B16(), Q0.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_DOT_PRODUCT_4, DOT_PRODUCT_4_V128);

// ============================================================================
// OPCODE_AND
// ============================================================================
// TODO(benvanik): put dest/src1|2 together.
template <typename SEQ, typename REG, typename ARGS>
void EmitAndXX(A64Emitter& e, const ARGS& i) {
  SEQ::EmitCommutativeBinaryOp(
      e, i,
      [](A64Emitter& e, REG dest_src, REG src) {
        e.AND(dest_src, dest_src, src);
      },
      [](A64Emitter& e, REG dest_src, int32_t constant) {
        e.MOV(REG(1), constant);
        e.AND(dest_src, dest_src, REG(1));
      });
}
struct AND_I8 : Sequence<AND_I8, I<OPCODE_AND, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAndXX<AND_I8, WReg>(e, i);
  }
};
struct AND_I16 : Sequence<AND_I16, I<OPCODE_AND, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAndXX<AND_I16, WReg>(e, i);
  }
};
struct AND_I32 : Sequence<AND_I32, I<OPCODE_AND, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAndXX<AND_I32, WReg>(e, i);
  }
};
struct AND_I64 : Sequence<AND_I64, I<OPCODE_AND, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAndXX<AND_I64, XReg>(e, i);
  }
};
struct AND_V128 : Sequence<AND_V128, I<OPCODE_AND, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp(
        e, i, [](A64Emitter& e, QReg dest, QReg src1, QReg src2) {
          e.AND(dest.B16(), src1.B16(), src2.B16());
        });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_AND, AND_I8, AND_I16, AND_I32, AND_I64, AND_V128);

// ============================================================================
// OPCODE_AND_NOT
// ============================================================================
template <typename SEQ, typename REG, typename ARGS>
void EmitAndNotXX(A64Emitter& e, const ARGS& i) {
  if (i.src1.is_constant) {
    // src1 constant.
    auto temp = GetTempReg<typename decltype(i.src1)::reg_type>(e);
    e.MOV(temp, i.src1.constant());
    e.BIC(i.dest, temp, i.src2);
  } else if (i.src2.is_constant) {
    // src2 constant.
    if (i.dest.reg().index() == i.src1.reg().index()) {
      auto temp = GetTempReg<typename decltype(i.src2)::reg_type>(e);
      e.MOV(temp, ~i.src2.constant());
      e.AND(i.dest, i.dest, temp);
    } else {
      e.MOV(i.dest, i.src1);
      auto temp = GetTempReg<typename decltype(i.src2)::reg_type>(e);
      e.MOV(temp, ~i.src2.constant());
      e.AND(i.dest, i.dest, temp);
    }
  } else {
    // neither are constant
    e.BIC(i.dest, i.src1, i.src2);
  }
}
struct AND_NOT_I8 : Sequence<AND_NOT_I8, I<OPCODE_AND_NOT, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAndNotXX<AND_NOT_I8, WReg>(e, i);
  }
};
struct AND_NOT_I16
    : Sequence<AND_NOT_I16, I<OPCODE_AND_NOT, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAndNotXX<AND_NOT_I16, WReg>(e, i);
  }
};
struct AND_NOT_I32
    : Sequence<AND_NOT_I32, I<OPCODE_AND_NOT, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAndNotXX<AND_NOT_I32, WReg>(e, i);
  }
};
struct AND_NOT_I64
    : Sequence<AND_NOT_I64, I<OPCODE_AND_NOT, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAndNotXX<AND_NOT_I64, XReg>(e, i);
  }
};
struct AND_NOT_V128
    : Sequence<AND_NOT_V128, I<OPCODE_AND_NOT, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp(
        e, i, [](A64Emitter& e, QReg dest, QReg src1, QReg src2) {
          e.BIC(dest.B16(), src1.B16(), src2.B16());
        });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_AND_NOT, AND_NOT_I8, AND_NOT_I16, AND_NOT_I32,
                     AND_NOT_I64, AND_NOT_V128);

// ============================================================================
// OPCODE_OR
// ============================================================================
// TODO(benvanik): put dest/src1|2 together.
template <typename SEQ, typename REG, typename ARGS>
void EmitOrXX(A64Emitter& e, const ARGS& i) {
  SEQ::EmitCommutativeBinaryOp(
      e, i,
      [](A64Emitter& e, REG dest_src, REG src) {
        e.ORR(dest_src, dest_src, src);
      },
      [](A64Emitter& e, REG dest_src, int32_t constant) {
        e.MOV(REG(1), constant);
        e.ORR(dest_src, dest_src, REG(1));
      });
}
struct OR_I8 : Sequence<OR_I8, I<OPCODE_OR, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitOrXX<OR_I8, WReg>(e, i);
  }
};
struct OR_I16 : Sequence<OR_I16, I<OPCODE_OR, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitOrXX<OR_I16, WReg>(e, i);
  }
};
struct OR_I32 : Sequence<OR_I32, I<OPCODE_OR, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitOrXX<OR_I32, WReg>(e, i);
  }
};
struct OR_I64 : Sequence<OR_I64, I<OPCODE_OR, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitOrXX<OR_I64, XReg>(e, i);
  }
};
struct OR_V128 : Sequence<OR_V128, I<OPCODE_OR, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp(
        e, i, [](A64Emitter& e, QReg dest, QReg src1, QReg src2) {
          e.ORR(dest.B16(), src1.B16(), src2.B16());
        });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_OR, OR_I8, OR_I16, OR_I32, OR_I64, OR_V128);

// ============================================================================
// OPCODE_XOR
// ============================================================================
// TODO(benvanik): put dest/src1|2 together.
template <typename SEQ, typename REG, typename ARGS>
void EmitXorXX(A64Emitter& e, const ARGS& i) {
  SEQ::EmitCommutativeBinaryOp(
      e, i,
      [](A64Emitter& e, REG dest_src, REG src) {
        e.EOR(dest_src, dest_src, src);
      },
      [](A64Emitter& e, REG dest_src, int32_t constant) {
        e.MOV(REG(1), constant);
        e.EOR(dest_src, dest_src, REG(1));
      });
}
struct XOR_I8 : Sequence<XOR_I8, I<OPCODE_XOR, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitXorXX<XOR_I8, WReg>(e, i);
  }
};
struct XOR_I16 : Sequence<XOR_I16, I<OPCODE_XOR, I16Op, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitXorXX<XOR_I16, WReg>(e, i);
  }
};
struct XOR_I32 : Sequence<XOR_I32, I<OPCODE_XOR, I32Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitXorXX<XOR_I32, WReg>(e, i);
  }
};
struct XOR_I64 : Sequence<XOR_I64, I<OPCODE_XOR, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitXorXX<XOR_I64, XReg>(e, i);
  }
};
struct XOR_V128 : Sequence<XOR_V128, I<OPCODE_XOR, V128Op, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitCommutativeBinaryVOp(
        e, i, [](A64Emitter& e, QReg dest, QReg src1, QReg src2) {
          e.EOR(dest.B16(), src1.B16(), src2.B16());
        });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_XOR, XOR_I8, XOR_I16, XOR_I32, XOR_I64, XOR_V128);

// ============================================================================
// OPCODE_NOT
// ============================================================================
// TODO(benvanik): put dest/src1 together.
template <typename SEQ, typename REG, typename ARGS>
void EmitNotXX(A64Emitter& e, const ARGS& i) {
  SEQ::EmitUnaryOp(
      e, i, [](A64Emitter& e, REG dest_src) { e.MVN(dest_src, dest_src); });
}
struct NOT_I8 : Sequence<NOT_I8, I<OPCODE_NOT, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitNotXX<NOT_I8, WReg>(e, i);
  }
};
struct NOT_I16 : Sequence<NOT_I16, I<OPCODE_NOT, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitNotXX<NOT_I16, WReg>(e, i);
  }
};
struct NOT_I32 : Sequence<NOT_I32, I<OPCODE_NOT, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitNotXX<NOT_I32, WReg>(e, i);
  }
};
struct NOT_I64 : Sequence<NOT_I64, I<OPCODE_NOT, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitNotXX<NOT_I64, XReg>(e, i);
  }
};
struct NOT_V128 : Sequence<NOT_V128, I<OPCODE_NOT, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    QReg src = i.src1.is_constant ? Q0 : i.src1.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    e.NOT(i.dest.reg().B16(), src.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_NOT, NOT_I8, NOT_I16, NOT_I32, NOT_I64, NOT_V128);

// ============================================================================
// OPCODE_SHL
// ============================================================================
// TODO(benvanik): optimize common shifts.
template <typename SEQ, typename REG, typename ARGS>
void EmitShlXX(A64Emitter& e, const ARGS& i) {
  SEQ::EmitAssociativeBinaryOp(
      e, i,
      [](A64Emitter& e, REG dest_src, WReg src) {
        e.LSL(dest_src, dest_src, REG(src.index()));
      },
      [](A64Emitter& e, REG dest_src, int8_t constant) {
        e.LSL(dest_src, dest_src, constant);
      });
}
struct SHL_I8 : Sequence<SHL_I8, I<OPCODE_SHL, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitShlXX<SHL_I8, WReg>(e, i);
  }
};
struct SHL_I16 : Sequence<SHL_I16, I<OPCODE_SHL, I16Op, I16Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitShlXX<SHL_I16, WReg>(e, i);
  }
};
struct SHL_I32 : Sequence<SHL_I32, I<OPCODE_SHL, I32Op, I32Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitShlXX<SHL_I32, WReg>(e, i);
  }
};
struct SHL_I64 : Sequence<SHL_I64, I<OPCODE_SHL, I64Op, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitShlXX<SHL_I64, XReg>(e, i);
  }
};
struct SHL_V128 : Sequence<SHL_V128, I<OPCODE_SHL, V128Op, V128Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // TODO(benvanik): native version (with shift magic).
    if (i.src2.is_constant) {
      e.MOV(e.GetNativeParam(1), i.src2.constant());
    } else {
      e.MOV(e.GetNativeParam(1), i.src2.reg().toX());
    }
    e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1));
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateShlV128));
    e.MOV(i.dest.reg().B16(), Q0.B16());
  }
  static float32x4_t EmulateShlV128(void*, std::byte src1[16], uint8_t src2) {
    // Almost all instances are shamt = 1, but non-constant.
    // shamt is [0,7]
    uint8_t shamt = src2 & 0x7;

    // Load `src1` as a byte vector (uint8x16_t) then work on byte shifting
    uint8x16_t byte_vec = vld1q_u8(reinterpret_cast<const uint8_t*>(src1));
    alignas(16) vec128_t value;
    vst1q_u8(reinterpret_cast<uint8_t*>(&value), byte_vec);
    for (int i = 0; i < 15; ++i) {
      value.u8[i ^ 0x3] = (value.u8[i ^ 0x3] << shamt) |
                          (value.u8[(i + 1) ^ 0x3] >> (8 - shamt));
    }
    value.u8[15 ^ 0x3] = value.u8[15 ^ 0x3] << shamt;
    return vreinterpretq_f32_u8(
        vld1q_u8(reinterpret_cast<const uint8_t*>(&value)));
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SHL, SHL_I8, SHL_I16, SHL_I32, SHL_I64, SHL_V128);

// ============================================================================
// OPCODE_SHR
// ============================================================================
struct SHR_I8 : Sequence<SHR_I8, I<OPCODE_SHR, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    Sequence::EmitAssociativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.LSR(dest_src, dest_src, src);
        },
        [](A64Emitter& e, WReg dest_src, int8_t constant) {
          e.LSR(dest_src, dest_src, constant);
        });
  }
};
struct SHR_I16 : Sequence<SHR_I16, I<OPCODE_SHR, I16Op, I16Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    Sequence::EmitAssociativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.LSR(dest_src, dest_src, src);
        },
        [](A64Emitter& e, WReg dest_src, int8_t constant) {
          e.LSR(dest_src, dest_src, constant);
        });
  }
};
struct SHR_I32 : Sequence<SHR_I32, I<OPCODE_SHR, I32Op, I32Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    Sequence::EmitAssociativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.LSR(dest_src, dest_src, src);
        },
        [](A64Emitter& e, WReg dest_src, int8_t constant) {
          e.LSR(dest_src, dest_src, constant);
        });
  }
};
struct SHR_I64 : Sequence<SHR_I64, I<OPCODE_SHR, I64Op, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    Sequence::EmitAssociativeBinaryOp(
        e, i,
        [](A64Emitter& e, XReg dest_src, WReg src) {
          e.LSR(dest_src, dest_src, src.toX());
        },
        [](A64Emitter& e, XReg dest_src, int8_t constant) {
          e.LSR(dest_src, dest_src, constant);
        });
  }
};
struct SHR_V128 : Sequence<SHR_V128, I<OPCODE_SHR, V128Op, V128Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // TODO(benvanik): native version (with shift magic).
    if (i.src2.is_constant) {
      e.MOV(e.GetNativeParam(1), i.src2.constant());
    } else {
      e.MOV(e.GetNativeParam(1), i.src2.reg().toX());
    }
    e.ADD(e.GetNativeParam(0), SP, e.StashV(0, i.src1));
    e.CallNativeSafe(reinterpret_cast<void*>(EmulateShrV128));
    e.MOV(i.dest.reg().B16(), Q0.B16());
  }
  static float32x4_t EmulateShrV128(void*, std::byte src1[16], uint8_t src2) {
    // Almost all instances are shamt = 1, but non-constant.
    // shamt is [0,7]
    uint8_t shamt = src2 & 0x7;
    // Load `src1` as a byte vector (uint8x16_t) and store it into `value`
    uint8x16_t byte_vec = vld1q_u8(reinterpret_cast<const uint8_t*>(src1));
    alignas(16) vec128_t value;
    vst1q_u8(reinterpret_cast<uint8_t*>(&value), byte_vec);

    for (int i = 15; i > 0; --i) {
      value.u8[i ^ 0x3] = (value.u8[i ^ 0x3] >> shamt) |
                          (value.u8[(i - 1) ^ 0x3] << (8 - shamt));
    }
    value.u8[0 ^ 0x3] = value.u8[0 ^ 0x3] >> shamt;
    // Convert to float32x4_t by reinterpreting the processed `value`
    return vreinterpretq_f32_u8(
        vld1q_u8(reinterpret_cast<const uint8_t*>(&value)));
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SHR, SHR_I8, SHR_I16, SHR_I32, SHR_I64, SHR_V128);

// ============================================================================
// OPCODE_SHA
// ============================================================================
struct SHA_I8 : Sequence<SHA_I8, I<OPCODE_SHA, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    Sequence::EmitAssociativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.SXTB(dest_src, dest_src);
          e.ASR(dest_src, dest_src, src);
        },
        [](A64Emitter& e, WReg dest_src, int8_t constant) {
          e.SXTB(dest_src, dest_src);
          e.ASR(dest_src, dest_src, constant);
        });
  }
};
struct SHA_I16 : Sequence<SHA_I16, I<OPCODE_SHA, I16Op, I16Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    Sequence::EmitAssociativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.SXTH(dest_src, dest_src);
          e.ASR(dest_src, dest_src, src);
        },
        [](A64Emitter& e, WReg dest_src, int8_t constant) {
          e.SXTH(dest_src, dest_src);
          e.ASR(dest_src, dest_src, constant);
        });
  }
};
struct SHA_I32 : Sequence<SHA_I32, I<OPCODE_SHA, I32Op, I32Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    Sequence::EmitAssociativeBinaryOp(
        e, i,
        [](A64Emitter& e, WReg dest_src, WReg src) {
          e.ASR(dest_src, dest_src, src);
        },
        [](A64Emitter& e, WReg dest_src, int8_t constant) {
          e.ASR(dest_src, dest_src, constant);
        });
  }
};
struct SHA_I64 : Sequence<SHA_I64, I<OPCODE_SHA, I64Op, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    Sequence::EmitAssociativeBinaryOp(
        e, i,
        [](A64Emitter& e, XReg dest_src, WReg src) {
          e.ASR(dest_src, dest_src, src.toX());
        },
        [](A64Emitter& e, XReg dest_src, int8_t constant) {
          e.ASR(dest_src, dest_src, constant);
        });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SHA, SHA_I8, SHA_I16, SHA_I32, SHA_I64);

// ============================================================================
// OPCODE_ROTATE_LEFT
// ============================================================================
// TODO(benvanik): put dest/src1 together, src2 in cl.
template <typename SEQ, typename ARGS>
void EmitRotateLeftNarrow(A64Emitter& e, const ARGS& i, uint32_t width) {
  const uint32_t value_mask = (uint32_t(1) << width) - 1;
  const uint32_t shift_mask = width - 1;

  if (i.src1.is_constant) {
    e.MOV(W0, uint32_t(i.src1.constant()));
  } else {
    e.MOV(W0, i.src1.reg().toW());
  }
  e.MOV(W4, value_mask);
  e.AND(W0, W0, W4);

  if (i.src2.is_constant) {
    e.MOV(W1, uint32_t(i.src2.constant()));
  } else {
    e.MOV(W1, i.src2.reg().toW());
  }
  e.MOV(W4, shift_mask);
  e.AND(W1, W1, W4);

  // left = (value << sh), right = (value >> (width - sh))
  e.MOV(W2, width);
  e.SUB(W2, W2, W1);
  e.LSLV(W3, W0, W1);
  e.LSRV(W1, W0, W2);
  e.ORR(W3, W3, W1);
  e.MOV(W4, value_mask);
  e.AND(i.dest, W3, W4);
}
struct ROTATE_LEFT_I8
    : Sequence<ROTATE_LEFT_I8, I<OPCODE_ROTATE_LEFT, I8Op, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitRotateLeftNarrow<ROTATE_LEFT_I8>(e, i, 8);
  }
};
struct ROTATE_LEFT_I16
    : Sequence<ROTATE_LEFT_I16, I<OPCODE_ROTATE_LEFT, I16Op, I16Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitRotateLeftNarrow<ROTATE_LEFT_I16>(e, i, 16);
  }
};
struct ROTATE_LEFT_I32
    : Sequence<ROTATE_LEFT_I32, I<OPCODE_ROTATE_LEFT, I32Op, I32Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.MOV(W0, i.src1.constant());
    } else {
      e.MOV(W0, i.src1.reg());
    }

    if (i.src2.is_constant) {
      e.MOV(W1, i.src2.constant());
    } else {
      e.SXTB(W1, i.src2.reg());
    }
    e.NEG(W1, W1);

    e.ROR(i.dest, W0, W1);
  }
};
struct ROTATE_LEFT_I64
    : Sequence<ROTATE_LEFT_I64, I<OPCODE_ROTATE_LEFT, I64Op, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src1.is_constant) {
      e.MOV(X0, i.src1.constant());
    } else {
      e.MOV(X0, i.src1.reg());
    }

    if (i.src2.is_constant) {
      e.MOV(X1, i.src2.constant());
    } else {
      e.SXTB(X1, i.src2.reg().toW());
    }
    e.NEG(X1, X1);

    e.ROR(i.dest, X0, X1);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ROTATE_LEFT, ROTATE_LEFT_I8, ROTATE_LEFT_I16,
                     ROTATE_LEFT_I32, ROTATE_LEFT_I64);

// ============================================================================
// OPCODE_BYTE_SWAP
// ============================================================================
// TODO(benvanik): put dest/src1 together.
struct BYTE_SWAP_I16
    : Sequence<BYTE_SWAP_I16, I<OPCODE_BYTE_SWAP, I16Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitUnaryOp(e, i, [](A64Emitter& e, WReg dest_src) {
      e.REV16(dest_src, dest_src);
    });
  }
};
struct BYTE_SWAP_I32
    : Sequence<BYTE_SWAP_I32, I<OPCODE_BYTE_SWAP, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitUnaryOp(
        e, i, [](A64Emitter& e, WReg dest_src) { e.REV(dest_src, dest_src); });
  }
};
struct BYTE_SWAP_I64
    : Sequence<BYTE_SWAP_I64, I<OPCODE_BYTE_SWAP, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitUnaryOp(
        e, i, [](A64Emitter& e, XReg dest_src) { e.REV(dest_src, dest_src); });
  }
};
struct BYTE_SWAP_V128
    : Sequence<BYTE_SWAP_V128, I<OPCODE_BYTE_SWAP, V128Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // Reverse upper and lower 64-bit halfs
    QReg src = i.src1.is_constant ? Q0 : i.src1.reg();
    if (i.src1.is_constant) {
      e.LoadConstantV(Q0, i.src1.constant());
    }
    e.REV32(i.dest.reg().B16(), src.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_BYTE_SWAP, BYTE_SWAP_I16, BYTE_SWAP_I32,
                     BYTE_SWAP_I64, BYTE_SWAP_V128);

// ============================================================================
// OPCODE_CNTLZ
// Count leading zeroes
// ============================================================================
struct CNTLZ_I8 : Sequence<CNTLZ_I8, I<OPCODE_CNTLZ, I8Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // No 8bit lzcnt, so do 32 and sub 24.
    e.UXTB(i.dest, i.src1);
    e.CLZ(i.dest, i.dest);
    e.SUB(i.dest.reg(), i.dest.reg(), 24);
  }
};
struct CNTLZ_I16 : Sequence<CNTLZ_I16, I<OPCODE_CNTLZ, I8Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // No 16bit lzcnt, so do 32 and sub 16.
    e.UXTH(i.dest, i.src1);
    e.CLZ(i.dest, i.dest);
    e.SUB(i.dest.reg(), i.dest.reg(), 16);
  }
};
struct CNTLZ_I32 : Sequence<CNTLZ_I32, I<OPCODE_CNTLZ, I8Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.CLZ(i.dest, i.src1);
  }
};
struct CNTLZ_I64 : Sequence<CNTLZ_I64, I<OPCODE_CNTLZ, I8Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.CLZ(i.dest.reg().toX(), i.src1);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CNTLZ, CNTLZ_I8, CNTLZ_I16, CNTLZ_I32, CNTLZ_I64);

// ============================================================================
// OPCODE_SET_ROUNDING_MODE
// ============================================================================
// Input: FPSCR (PPC format)
// Convert from PPC rounding mode to ARM
// PPC | ARM |
// 00  | 00  | nearest
// 01  | 11  | toward zero
// 10  | 01  | toward +infinity
// 11  | 10  | toward -infinity
static const uint8_t fpcr_table[] = {
    0b0'00,  // |--|nearest
    0b0'11,  // |--|toward zero
    0b0'01,  // |--|toward +infinity
    0b0'10,  // |--|toward -infinity
    0b1'00,  // |FZ|nearest
    0b1'11,  // |FZ|toward zero
    0b1'01,  // |FZ|toward +infinity
    0b1'10,  // |FZ|toward -infinity
};

uint64_t SetNonIEEEModeForwarder(void* raw_context, uint64_t control) {
  control &= 0b111;
  auto* backend_context = reinterpret_cast<A64BackendContext*>(
      reinterpret_cast<std::byte*>(raw_context) - sizeof(A64BackendContext));
  backend_context->non_ieee_mode = (control >> 2) & 1;
  return control;
}

struct SET_ROUNDING_MODE_I32
    : Sequence<SET_ROUNDING_MODE_I32,
               I<OPCODE_SET_ROUNDING_MODE, VoidOp, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // Low 3 bits are |Non-IEEE:1|RoundingMode:2|
    // Non-IEEE bit is flush-to-zero
    if (i.src1.is_constant) {
      const uint32_t control = static_cast<uint32_t>(i.src1.constant()) & 0b111;
      e.CallNative(SetNonIEEEModeForwarder, control);
      e.MOV(W1, control);
    } else {
      e.AND(W1, i.src1.reg(), 0b111);
      e.CallNativeSafe(reinterpret_cast<void*>(SetNonIEEEModeForwarder));
      e.MOV(W1, W0);
    }

    // Use the low 3 bits as an index into a LUT
    e.MOV(X0, reinterpret_cast<uintptr_t>(fpcr_table));
    e.LDRB(W0, X0, X1);

    // Replace FPCR bits with new value
    e.MRS(X1, SystemReg::FPCR);
    e.BFI(X1, X0, 23, 3);
    e.MSR(SystemReg::FPCR, X1);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SET_ROUNDING_MODE, SET_ROUNDING_MODE_I32);

static void MaybeYieldForwarder(void* ctx) { xe::threading::MaybeYield(); }
// ============================================================================
// OPCODE_DELAY_EXECUTION
// ============================================================================
struct DELAY_EXECUTION
    : Sequence<DELAY_EXECUTION, I<OPCODE_DELAY_EXECUTION, VoidOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.CallNativeSafe(reinterpret_cast<void*>(MaybeYieldForwarder));
  }
};
EMITTER_OPCODE_TABLE(OPCODE_DELAY_EXECUTION, DELAY_EXECUTION);

// Include anchors to other sequence sources so they get included in the build.
extern volatile int anchor_control;
static int anchor_control_dest = anchor_control;

extern volatile int anchor_memory;
static int anchor_memory_dest = anchor_memory;

extern volatile int anchor_vector;
static int anchor_vector_dest = anchor_vector;

bool SelectSequence(A64Emitter* e, const hir::Instr* i,
                    const hir::Instr** new_tail) {
  const InstrKey key(i);
  auto it = GetSequenceTable().find(key);
  if (it != GetSequenceTable().end()) {
    if (it->second(*e, i)) {
      *new_tail = i->next;
      return true;
    }
  }
  XELOGE("No sequence match for variant {}", hir::GetOpcodeName(i->opcode));
  return false;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
