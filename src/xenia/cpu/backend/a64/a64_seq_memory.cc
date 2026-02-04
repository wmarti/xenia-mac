/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Xenia Developers. All rights reserved.                      *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_sequences.h"

#include <algorithm>
#include <atomic>
#include <cstring>

#include "xenia/base/byte_order.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/cpu/backend/a64/a64_op.h"
#include "xenia/cpu/backend/a64/a64_tracers.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/xex_module.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/memory.h"

DECLARE_bool(emit_mmio_aware_stores_for_recorded_exception_addresses);
DEFINE_bool(a64_force_mmio_aware_byteswap_loads, false,
            "Force MMIO-aware helper loads for byte-swapped I32 loads.",
            "CPU");
DEFINE_bool(a64_log_reservation_failures, false,
            "Log failed A64 reservation stores (ldarx/stdcx).", "CPU");
DEFINE_uint32(a64_reservation_log_rate, 4096,
              "Log 1 in N failed A64 reservation stores (0 = log all).",
              "CPU");
DEFINE_uint32(a64_reservation_watch_address, 0,
              "If non-zero, only log failed A64 reservation stores for this "
              "guest address.",
              "CPU");
DEFINE_uint32(a64_watch_store_address, 0,
              "If non-zero, log 32-bit guest stores to this address.",
              "CPU");

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

volatile int anchor_memory = 0;

static bool IsPossibleMMIOInstruction(A64Emitter& e, const hir::Instr* i) {
  if (!cvars::emit_mmio_aware_stores_for_recorded_exception_addresses) {
    return false;
  }
  if (IsTracingData()) {
    return false;
  }
  uint32_t guestaddr = i->GuestAddressFor();
  if (!guestaddr) {
    return false;
  }

  auto guest_module = e.GuestModule();
  if (!guest_module) {
    return false;
  }
  auto flags = guest_module->GetInstructionAddressFlags(guestaddr);
  return flags && flags->accessed_mmio;
}

template <typename T, bool swap>
static void MMIOAwareStore(void* _ctx, unsigned int guestaddr, T value) {
  if (swap) {
    value = xe::byte_swap(value);
  }
  if (guestaddr >= 0xE0000000) {
    guestaddr += 0x1000;
  }

  auto ctx = reinterpret_cast<ppc::PPCContext*>(_ctx);
  auto gaddr = ctx->processor->memory()->LookupVirtualMappedRange(guestaddr);
  if (!gaddr) {
    *reinterpret_cast<T*>(ctx->virtual_membase + guestaddr) = value;
  } else {
    value = xe::byte_swap(value);
    gaddr->write(nullptr, gaddr->callback_context, guestaddr, value);
  }
}

template <typename T, bool swap>
static T MMIOAwareLoad(void* _ctx, unsigned int guestaddr) {
  T value;

  if (guestaddr >= 0xE0000000) {
    guestaddr += 0x1000;
  }

  auto ctx = reinterpret_cast<ppc::PPCContext*>(_ctx);
  auto gaddr = ctx->processor->memory()->LookupVirtualMappedRange(guestaddr);
  if (!gaddr) {
    value = *reinterpret_cast<T*>(ctx->virtual_membase + guestaddr);
    if (swap) {
      value = xe::byte_swap(value);
    }
  } else {
    value = gaddr->read(nullptr, gaddr->callback_context, guestaddr);
  }
  return value;
}

static void LogReservationStore32(void* raw_context, uint64_t guest_addr,
                                  uint64_t value, uint64_t status) {
  if (!cvars::a64_log_reservation_failures) {
    return;
  }
  if (status == 0) {
    return;
  }
  uint32_t watch = cvars::a64_reservation_watch_address;
  if (watch && uint32_t(guest_addr) != watch) {
    return;
  }
  uint32_t rate = cvars::a64_reservation_log_rate;
  if (rate) {
    static std::atomic<uint32_t> log_count{0};
    if ((log_count.fetch_add(1) % rate) != 0) {
      return;
    }
  }
  uint32_t thread_id = 0;
  if (raw_context) {
    thread_id = reinterpret_cast<ppc::PPCContext*>(raw_context)->thread_id;
  }
  XELOGI(
      "A64 reservation store failed: guest=0x{:08X} size=4 value=0x{:08X} "
      "status=0x{:X} tid=0x{:08X}",
      uint32_t(guest_addr), uint32_t(value), uint32_t(status), thread_id);
}

static void LogStoreWatch32(void* raw_context, uint64_t guest_addr,
                            uint64_t value, uint64_t guest_pc) {
  uint32_t thread_id = 0;
  const ppc::PPCContext* context =
      reinterpret_cast<ppc::PPCContext*>(raw_context);
  if (context) {
    thread_id = context->thread_id;
  }
  XELOGI(
      "A64 store watch: guest_pc=0x{:08X} addr=0x{:08X} value=0x{:08X} "
      "tid=0x{:08X}",
      uint32_t(guest_pc), uint32_t(guest_addr), uint32_t(value), thread_id);
  if (context && uint32_t(value) == 0) {
    static std::atomic<bool> logged_dump{false};
    const uint32_t r11 = uint32_t(context->r[11]);
    const uint32_t r31 = uint32_t(context->r[31]);
    uint32_t raw = 0;
    uint32_t prot = 0;
    bool read_ok = false;
    if (context->kernel_state && context->kernel_state->memory()) {
      auto* memory = context->kernel_state->memory();
      auto* heap = memory->LookupHeap(r11);
      if (heap && heap->QueryProtect(r11, &prot) &&
          (prot & kMemoryProtectRead)) {
        std::memcpy(&raw, memory->TranslateVirtual(r11), sizeof(raw));
        read_ok = true;
      }
    }
    if (read_ok) {
      XELOGI(
          "A64 store watch zero: guest_pc=0x{:08X} r30=0x{:08X} "
          "r31=0x{:08X} r9=0x{:08X} r10=0x{:08X} r11=0x{:08X} "
          "lr=0x{:08X} [r11]=0x{:08X} be=0x{:08X} prot=0x{:X}",
          uint32_t(guest_pc), uint32_t(context->r[30]),
          uint32_t(context->r[31]), uint32_t(context->r[9]),
          uint32_t(context->r[10]), r11, uint32_t(context->lr), raw,
          xe::byte_swap(raw), prot);
    } else {
      XELOGI(
          "A64 store watch zero: guest_pc=0x{:08X} r30=0x{:08X} "
          "r31=0x{:08X} r9=0x{:08X} r10=0x{:08X} r11=0x{:08X} "
          "lr=0x{:08X} [r11]=<unreadable>",
          uint32_t(guest_pc), uint32_t(context->r[30]),
          uint32_t(context->r[31]), uint32_t(context->r[9]),
          uint32_t(context->r[10]), r11, uint32_t(context->lr));
    }
    if (context->kernel_state && context->kernel_state->memory() &&
        !logged_dump.exchange(true)) {
      auto* memory = context->kernel_state->memory();
      auto* heap = memory->LookupHeap(r31);
      uint32_t dump_words[16] = {};
      bool dump_ok = false;
      if (heap && heap->QueryProtect(r31, &prot) &&
          (prot & kMemoryProtectRead)) {
        std::memcpy(dump_words, memory->TranslateVirtual(r31),
                    sizeof(dump_words));
        dump_ok = true;
      }
      if (dump_ok) {
        XELOGI(
            "A64 store watch zero dump r31=0x{:08X}: "
            "{:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} "
            "{:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
            r31, dump_words[0], dump_words[1], dump_words[2], dump_words[3],
            dump_words[4], dump_words[5], dump_words[6], dump_words[7],
            dump_words[8], dump_words[9], dump_words[10], dump_words[11],
            dump_words[12], dump_words[13], dump_words[14], dump_words[15]);
      } else {
        XELOGI("A64 store watch zero dump r31=0x{:08X}: <unreadable>", r31);
      }
    }
  }
}

static void LogStoreWatch64(void* raw_context, uint64_t guest_addr,
                            uint64_t value, uint64_t guest_pc) {
  uint32_t thread_id = 0;
  const ppc::PPCContext* context =
      reinterpret_cast<ppc::PPCContext*>(raw_context);
  if (context) {
    thread_id = context->thread_id;
  }
  uint32_t watch_addr = cvars::a64_watch_store_address;
  uint32_t addr32 = uint32_t(guest_addr);
  const uint32_t low = uint32_t(value);
  const uint32_t high = uint32_t(value >> 32);
  const char* which = "base";
  if (watch_addr && addr32 + 4 == watch_addr) {
    which = "base+4";
  }
  XELOGI(
      "A64 store watch64: guest_pc=0x{:08X} addr=0x{:08X} {} "
      "value=0x{:016X} low=0x{:08X} high=0x{:08X} tid=0x{:08X}",
      uint32_t(guest_pc), addr32, which, value, low, high, thread_id);
}

static void LogReservationStore64(void* raw_context, uint64_t guest_addr,
                                  uint64_t value, uint64_t status) {
  if (!cvars::a64_log_reservation_failures) {
    return;
  }
  if (status == 0) {
    return;
  }
  uint32_t watch = cvars::a64_reservation_watch_address;
  if (watch && uint32_t(guest_addr) != watch) {
    return;
  }
  uint32_t rate = cvars::a64_reservation_log_rate;
  if (rate) {
    static std::atomic<uint32_t> log_count{0};
    if ((log_count.fetch_add(1) % rate) != 0) {
      return;
    }
  }
  uint32_t thread_id = 0;
  if (raw_context) {
    thread_id = reinterpret_cast<ppc::PPCContext*>(raw_context)->thread_id;
  }
  XELOGI(
      "A64 reservation store failed: guest=0x{:08X} size=8 value=0x{:016X} "
      "status=0x{:X} tid=0x{:08X}",
      uint32_t(guest_addr), value, uint32_t(status), thread_id);
}

// vec128b stores bytes in reversed 32-bit chunks; use reversed args for 0..15.
static const vec128_t kStvlShuffle =
    vec128b(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
static const vec128_t kStvrSwapMask = vec128b(static_cast<uint8_t>(0x83));

template <typename T>
XReg ComputeMemoryAddressOffset(A64Emitter& e, const T& guest, const T& offset,
                                WReg address_register = W3) {
  assert_true(offset.is_constant);
  const int32_t offset_const = static_cast<int32_t>(offset.constant());

  if (guest.is_constant) {
    uint32_t address = static_cast<uint32_t>(guest.constant());
    address += offset_const;
    if (address < 0x80000000) {
      e.MOV(address_register.toX(), address);
      e.ADD(address_register.toX(), e.GetMembaseReg(), address_register.toX());
      return address_register.toX();
    } else {
      if (address >= 0xE0000000 &&
          xe::memory::allocation_granularity() > 0x1000) {
        e.MOV(W0, address + 0x1000);
      } else {
        e.MOV(W0, address);
      }
      e.ADD(address_register.toX(), e.GetMembaseReg(), X0);
      return address_register.toX();
    }
  } else {
    if (xe::memory::allocation_granularity() > 0x1000) {
      // Emulate the 4 KB physical address offset in 0xE0000000+ when can't do
      // it via memory mapping.
      e.MOV(W0, 0xE0000000 - offset_const);
      e.CMP(guest.reg().toW(), W0);
      e.CSET(W0, Cond::HS);
      e.ADD(W0, guest.reg().toW(), W0, LSL, 12);
    } else {
      // Clear the top 32 bits, as they are likely garbage.
      // TODO(benvanik): find a way to avoid doing this.
      e.MOV(W0, guest.reg().toW());
    }
    // Guest addresses are 32-bit and wrap on addition.
    e.MOV(W1, offset_const);
    e.ADD(W0, W0, W1);

    e.ADD(address_register.toX(), e.GetMembaseReg(), X0);
    return address_register.toX();
  }
}

// Note: most *should* be aligned, but needs to be checked!
template <typename T>
XReg ComputeMemoryAddress(A64Emitter& e, const T& guest,
                          WReg address_register = W3) {
  if (guest.is_constant) {
    // TODO(benvanik): figure out how to do this without a temp.
    // Since the constant is often 0x8... if we tried to use that as a
    // displacement it would be sign extended and mess things up.
    const uint32_t address = static_cast<uint32_t>(guest.constant());
    if (address < 0x80000000) {
      e.MOV(W0, address);
      e.ADD(address_register.toX(), e.GetMembaseReg(), X0);
      return address_register.toX();
    } else {
      if (address >= 0xE0000000 &&
          xe::memory::allocation_granularity() > 0x1000) {
        e.MOV(W0, address + 0x1000u);
      } else {
        e.MOV(W0, address);
      }
      e.ADD(address_register.toX(), e.GetMembaseReg(), X0);
      return address_register.toX();
    }
  } else {
    if (xe::memory::allocation_granularity() > 0x1000) {
      // Emulate the 4 KB physical address offset in 0xE0000000+ when can't do
      // it via memory mapping.
      e.MOV(W0, 0xE0000000);
      e.CMP(guest.reg().toW(), W0);
      e.CSET(W0, Cond::HS);
      e.ADD(W0, guest.reg().toW(), W0, LSL, 12);
    } else {
      // Clear the top 32 bits, as they are likely garbage.
      // TODO(benvanik): find a way to avoid doing this.
      e.MOV(W0, guest.reg().toW());
    }
    e.ADD(address_register.toX(), e.GetMembaseReg(), X0);
    return address_register.toX();
  }
}

// ============================================================================
// OPCODE_ATOMIC_EXCHANGE
// ============================================================================
// Note that the address we use here is a real, host address!
// This is weird, and should be fixed.
static void EmitAtomicExchangeFallbackI8(A64Emitter& e, WReg dest,
                                         XReg address) {
  oaknut::Label retry;
  e.l(retry);
  e.LDAXRB(W4, address);
  e.STLXRB(W5, dest, address);
  e.CBNZ(W5, retry);
  e.MOV(dest, W4);
  e.UXTB(dest, dest);
}

static void EmitAtomicExchangeFallbackI16(A64Emitter& e, WReg dest,
                                          XReg address) {
  oaknut::Label retry;
  e.l(retry);
  e.LDAXRH(W4, address);
  e.STLXRH(W5, dest, address);
  e.CBNZ(W5, retry);
  e.MOV(dest, W4);
  e.UXTH(dest, dest);
}

static void EmitAtomicExchangeFallbackI32(A64Emitter& e, WReg dest,
                                          XReg address) {
  oaknut::Label retry;
  e.l(retry);
  e.LDAXR(W4, address);
  e.STLXR(W5, dest, address);
  e.CBNZ(W5, retry);
  e.MOV(dest, W4);
}

static void EmitAtomicExchangeFallbackI64(A64Emitter& e, XReg dest,
                                          XReg address) {
  oaknut::Label retry;
  e.l(retry);
  e.LDAXR(X4, address);
  e.STLXR(W5, dest, address);
  e.CBNZ(W5, retry);
  e.MOV(dest, X4);
}

template <typename SEQ, typename REG, typename ARGS, typename FN_LSE,
          typename FN_FALLBACK>
void EmitAtomicExchangeXX(A64Emitter& e, const ARGS& i,
                          const FN_LSE& lse_fn,
                          const FN_FALLBACK& fallback_fn) {
  auto emit_exchange = [&](const XReg& address_reg) {
    if (e.IsFeatureEnabled(kA64EmitLSE)) {
      lse_fn(e, i.dest, address_reg);
    } else {
      fallback_fn(e, i.dest, address_reg);
    }
  };
  if (i.dest == i.src1) {
    e.MOV(X0, i.src1);
    if (i.dest != i.src2) {
      if (i.src2.is_constant) {
        e.MOV(i.dest, i.src2.constant());
      } else {
        e.MOV(i.dest, i.src2);
      }
    }
    emit_exchange(X0);
  } else {
    if (i.dest != i.src2) {
      if (i.src2.is_constant) {
        e.MOV(i.dest, i.src2.constant());
      } else {
        e.MOV(i.dest, i.src2);
      }
    }
    emit_exchange(i.src1);
  }
}
struct ATOMIC_EXCHANGE_I8
    : Sequence<ATOMIC_EXCHANGE_I8,
               I<OPCODE_ATOMIC_EXCHANGE, I8Op, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAtomicExchangeXX<ATOMIC_EXCHANGE_I8, WReg>(
        e, i,
        [](A64Emitter& e, WReg dest, XReg src) { e.SWPALB(dest, dest, src); },
        [](A64Emitter& e, WReg dest, XReg address) {
          EmitAtomicExchangeFallbackI8(e, dest, address);
        });
  }
};
struct ATOMIC_EXCHANGE_I16
    : Sequence<ATOMIC_EXCHANGE_I16,
               I<OPCODE_ATOMIC_EXCHANGE, I16Op, I64Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAtomicExchangeXX<ATOMIC_EXCHANGE_I8, WReg>(
        e, i,
        [](A64Emitter& e, WReg dest, XReg src) { e.SWPALH(dest, dest, src); },
        [](A64Emitter& e, WReg dest, XReg address) {
          EmitAtomicExchangeFallbackI16(e, dest, address);
        });
  }
};
struct ATOMIC_EXCHANGE_I32
    : Sequence<ATOMIC_EXCHANGE_I32,
               I<OPCODE_ATOMIC_EXCHANGE, I32Op, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAtomicExchangeXX<ATOMIC_EXCHANGE_I8, WReg>(
        e, i,
        [](A64Emitter& e, WReg dest, XReg src) { e.SWPAL(dest, dest, src); },
        [](A64Emitter& e, WReg dest, XReg address) {
          EmitAtomicExchangeFallbackI32(e, dest, address);
        });
  }
};
struct ATOMIC_EXCHANGE_I64
    : Sequence<ATOMIC_EXCHANGE_I64,
               I<OPCODE_ATOMIC_EXCHANGE, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitAtomicExchangeXX<ATOMIC_EXCHANGE_I8, XReg>(
        e, i,
        [](A64Emitter& e, XReg dest, XReg src) { e.SWPAL(dest, dest, src); },
        [](A64Emitter& e, XReg dest, XReg address) {
          EmitAtomicExchangeFallbackI64(e, dest, address);
        });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ATOMIC_EXCHANGE, ATOMIC_EXCHANGE_I8,
                     ATOMIC_EXCHANGE_I16, ATOMIC_EXCHANGE_I32,
                     ATOMIC_EXCHANGE_I64);

// ============================================================================
// OPCODE_LVL/LVR/STVL/STVR
// ============================================================================
struct LVL_V128 : Sequence<LVL_V128, I<OPCODE_LVL, V128Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const XReg address = ComputeMemoryAddress(e, i.src1, W4);
    e.AND(W0, address.toW(), 0xF);
    e.SUB(X1, address, X0);

    e.LDR(Q2, X1);

    e.MOV(X2, e.GetVConstPtr());
    e.LDR(Q0, X2, e.GetVConstOffset(VByteSwapMask));
    e.DUP(Q1.B16(), W0);
    e.ADD(Q0.B16(), Q0.B16(), Q1.B16());
    e.TBL(i.dest.reg().B16(), List{Q2.B16()}, Q0.B16());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LVL, LVL_V128);

struct LVR_V128 : Sequence<LVR_V128, I<OPCODE_LVR, V128Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const XReg address = ComputeMemoryAddress(e, i.src1, W4);
    e.AND(W0, address.toW(), 0xF);
    e.EOR(i.dest.reg().B16(), i.dest.reg().B16(), i.dest.reg().B16());

    oaknut::Label done;
    e.CBZ(W0, done);

    e.SUB(X1, address, X0);
    e.LDR(Q2, X1);

    e.MOV(X2, e.GetVConstPtr());
    e.LDR(Q0, X2, e.GetVConstOffset(VByteSwapMask));
    e.DUP(Q1.B16(), W0);
    e.ADD(Q0.B16(), Q0.B16(), Q1.B16());

    e.MOVI(Q1.B16(), 0x10);
    e.CMHS(Q3.B16(), Q0.B16(), Q1.B16());
    e.SUB(Q0.B16(), Q0.B16(), Q1.B16());
    e.MOVI(Q1.B16(), 0x80);
    e.BSL(Q3.B16(), Q0.B16(), Q1.B16());

    e.TBL(i.dest.reg().B16(), List{Q2.B16()}, Q3.B16());
    e.l(done);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LVR, LVR_V128);

struct STVL_V128 : Sequence<STVL_V128, I<OPCODE_STVL, VoidOp, I64Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const XReg address = ComputeMemoryAddress(e, i.src1, W4);
    e.AND(W0, address.toW(), 0xF);
    e.SUB(X1, address, X0);

    e.LDR(Q2, X1);

    e.MOV(X2, reinterpret_cast<uintptr_t>(&kStvlShuffle));
    e.LDR(Q0, X2);
    e.DUP(Q1.B16(), W0);
    e.SUB(Q0.B16(), Q0.B16(), Q1.B16());

    e.MOV(X2, e.GetVConstPtr());
    e.LDR(Q1, X2, e.GetVConstOffset(VSwapWordMask));
    e.EOR(Q0.B16(), Q0.B16(), Q1.B16());

    const QReg shuffled = Q3;
    if (i.src2.is_constant) {
      e.LoadConstantV(shuffled, i.src2.constant());
    } else {
      e.MOV(shuffled.B16(), i.src2.reg().B16());
    }
    e.TBL(shuffled.B16(), List{shuffled.B16()}, Q0.B16());

    e.MOVI(Q1.B16(), 0x80);
    e.CMHS(Q1.B16(), Q0.B16(), Q1.B16());
    e.BSL(Q1.B16(), Q2.B16(), shuffled.B16());
    e.STR(Q1, X1);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STVL, STVL_V128);

struct STVR_V128 : Sequence<STVR_V128, I<OPCODE_STVR, VoidOp, I64Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const XReg address = ComputeMemoryAddress(e, i.src1, W4);
    e.AND(W0, address.toW(), 0xF);

    oaknut::Label done;
    e.CBZ(W0, done);

    e.SUB(X1, address, X0);
    e.LDR(Q2, X1);

    e.MOV(X2, reinterpret_cast<uintptr_t>(&kStvlShuffle));
    e.LDR(Q0, X2);
    e.DUP(Q1.B16(), W0);
    e.SUB(Q0.B16(), Q0.B16(), Q1.B16());

    e.MOV(X2, reinterpret_cast<uintptr_t>(&kStvrSwapMask));
    e.LDR(Q1, X2);
    e.EOR(Q0.B16(), Q0.B16(), Q1.B16());

    e.MOVI(Q1.B16(), 0x0F);
    e.AND(Q1.B16(), Q0.B16(), Q1.B16());
    e.MOVI(Q3.B16(), 0x80);
    e.AND(Q3.B16(), Q0.B16(), Q3.B16());
    e.ORR(Q1.B16(), Q1.B16(), Q3.B16());

    const QReg shuffled = Q3;
    if (i.src2.is_constant) {
      e.LoadConstantV(shuffled, i.src2.constant());
    } else {
      e.MOV(shuffled.B16(), i.src2.reg().B16());
    }
    e.TBL(shuffled.B16(), List{shuffled.B16()}, Q1.B16());

    e.MOVI(Q1.B16(), 0x80);
    e.CMHS(Q1.B16(), Q0.B16(), Q1.B16());
    e.BSL(Q1.B16(), Q2.B16(), shuffled.B16());
    e.STR(Q1, X1);

    e.l(done);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STVR, STVR_V128);

// ============================================================================
// OPCODE_ATOMIC_COMPARE_EXCHANGE
// ============================================================================
struct ATOMIC_COMPARE_EXCHANGE_I32
    : Sequence<ATOMIC_COMPARE_EXCHANGE_I32,
               I<OPCODE_ATOMIC_COMPARE_EXCHANGE, I8Op, I64Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (xe::memory::allocation_granularity() > 0x1000) {
      // Emulate the 4 KB physical address offset in 0xE0000000+ when can't do
      // it via memory mapping.
      e.MOV(W3, 0xE0000000);
      e.CMP(i.src1.reg().toW(), W3);
      e.CSET(W1, Cond::HS);
      e.ADD(W1, i.src1.reg().toW(), W1, LSL, 12);
    } else {
      e.MOV(W1, i.src1.reg().toW());
    }
    e.ADD(X1, e.GetMembaseReg(), X1);

    const XReg address = X1;
    const WReg expected = i.src2;
    const WReg desired = i.src3;
    const WReg status = W0;

    if (e.IsFeatureEnabled(kA64EmitLSE)) {
      e.MOV(status, expected);

      // if([C] == A) [C] = B
      // else A = [C]
      e.CASAL(status, desired, address);
      e.CMP(status, expected);
      e.CSET(i.dest, Cond::EQ);
      return;
    }

    oaknut::Label success, fail, retry;

    e.l(retry);
    e.LDAXR(W4, address);
    e.CMP(W4, expected);
    e.B(Cond::NE, fail);

    e.STLXR(status.toW(), desired, address);
    e.CBNZ(status, retry);
    e.B(success);

    e.l(fail);
    e.CLREX();

    e.l(success);
    e.CSET(i.dest, Cond::EQ);
  }
};
struct ATOMIC_COMPARE_EXCHANGE_I64
    : Sequence<ATOMIC_COMPARE_EXCHANGE_I64,
               I<OPCODE_ATOMIC_COMPARE_EXCHANGE, I8Op, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (xe::memory::allocation_granularity() > 0x1000) {
      // Emulate the 4 KB physical address offset in 0xE0000000+ when can't do
      // it via memory mapping.
      e.MOV(W3, 0xE0000000);
      e.CMP(i.src1.reg(), X3);
      e.CSET(W1, Cond::HS);
      e.ADD(W1, i.src1.reg().toW(), W1, LSL, 12);
    } else {
      e.MOV(W1, i.src1.reg().toW());
    }
    e.ADD(X1, e.GetMembaseReg(), X1);

    const XReg address = X1;
    const XReg expected = i.src2;
    const XReg desired = i.src3;
    const XReg status = X0;

    if (e.IsFeatureEnabled(kA64EmitLSE)) {
      e.MOV(status, expected);

      // if([C] == A) [C] = B
      // else A = [C]
      e.CASAL(status, desired, address);
      e.CMP(status, expected);
      e.CSET(i.dest, Cond::EQ);
      return;
    }

    oaknut::Label success, fail, retry;

    e.l(retry);
    e.LDAXR(X4, address);
    e.CMP(X4, expected);
    e.B(Cond::NE, fail);

    e.STLXR(status.toW(), desired, address);
    e.CBNZ(status, retry);
    e.B(success);

    e.l(fail);
    e.CLREX();

    e.l(success);
    e.CSET(i.dest, Cond::EQ);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ATOMIC_COMPARE_EXCHANGE,
                     ATOMIC_COMPARE_EXCHANGE_I32, ATOMIC_COMPARE_EXCHANGE_I64);

// ============================================================================
// OPCODE_RESERVED_LOAD / OPCODE_RESERVED_STORE
// ============================================================================
struct RESERVED_LOAD_INT32
    : Sequence<RESERVED_LOAD_INT32, I<OPCODE_RESERVED_LOAD, I32Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const XReg address = ComputeMemoryAddress(e, i.src1, W3);
    e.LDAXR(i.dest, address);
  }
};
struct RESERVED_LOAD_INT64
    : Sequence<RESERVED_LOAD_INT64, I<OPCODE_RESERVED_LOAD, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const XReg address = ComputeMemoryAddress(e, i.src1, W3);
    e.LDAXR(i.dest, address);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_RESERVED_LOAD, RESERVED_LOAD_INT32,
                     RESERVED_LOAD_INT64);

struct RESERVED_STORE_INT32
    : Sequence<RESERVED_STORE_INT32,
               I<OPCODE_RESERVED_STORE, I8Op, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const XReg address = ComputeMemoryAddress(e, i.src1, W3);
    const WReg value = i.src2.is_constant ? W4 : i.src2;
    if (i.src2.is_constant) {
      e.MOV(value, static_cast<uint32_t>(i.src2.constant()));
    }
    e.STLXR(W0, value, address);
    const bool emit_log = cvars::a64_log_reservation_failures;
    if (emit_log) {
      const WReg status = W5;
      e.MOV(status, W0);
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toW();
      const auto status_reg = e.GetNativeParam(2).toW();
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(value_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(value_reg, value);
      }
      e.MOV(status_reg, status);
      e.CallNativeSafe(reinterpret_cast<void*>(LogReservationStore32));
      e.CMP(status, 0);
    } else {
      e.CMP(W0, 0);
    }
    e.CSET(i.dest, Cond::EQ);
  }
};

struct RESERVED_STORE_INT64
    : Sequence<RESERVED_STORE_INT64,
               I<OPCODE_RESERVED_STORE, I8Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const XReg address = ComputeMemoryAddress(e, i.src1, W3);
    const XReg value = i.src2.is_constant ? X4 : i.src2;
    if (i.src2.is_constant) {
      e.MOV(value, i.src2.constant());
    }
    e.STLXR(W0, value, address);
    const bool emit_log = cvars::a64_log_reservation_failures;
    if (emit_log) {
      const WReg status = W5;
      e.MOV(status, W0);
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1);
      const auto status_reg = e.GetNativeParam(2).toW();
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(value_reg, i.src2.constant());
      } else {
        e.MOV(value_reg, value);
      }
      e.MOV(status_reg, status);
      e.CallNativeSafe(reinterpret_cast<void*>(LogReservationStore64));
      e.CMP(status, 0);
    } else {
      e.CMP(W0, 0);
    }
    e.CSET(i.dest, Cond::EQ);
  }
};

EMITTER_OPCODE_TABLE(OPCODE_RESERVED_STORE, RESERVED_STORE_INT32,
                     RESERVED_STORE_INT64);

// ============================================================================
// OPCODE_LOAD_LOCAL
// ============================================================================
// Note: all types are always aligned on the stack.
template <typename EmitFn>
void EmitLocalAccess(A64Emitter& e, uint32_t offset, uint32_t scale,
                     const EmitFn& emit_fn) {
  const bool imm_valid = (offset % scale) == 0 && (offset / scale) <= 0xFFF;
  if (imm_valid) {
    emit_fn(SP, offset);
    return;
  }

  auto addr = GetTempReg<XReg>(e);
  e.MOV(addr, offset);
  e.ADD(addr, SP, addr);
  emit_fn(addr, 0);
}

struct LOAD_LOCAL_I8
    : Sequence<LOAD_LOCAL_I8, I<OPCODE_LOAD_LOCAL, I8Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitLocalAccess(e, i.src1.constant(), 1, [&](auto base, uint32_t imm) {
      e.LDRB(i.dest, base, imm);
    });
    // e.TraceLoadI8(DATA_LOCAL, i.src1.constant, i.dest);
  }
};
struct LOAD_LOCAL_I16
    : Sequence<LOAD_LOCAL_I16, I<OPCODE_LOAD_LOCAL, I16Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitLocalAccess(e, i.src1.constant(), 2, [&](auto base, uint32_t imm) {
      e.LDRH(i.dest, base, imm);
    });
    // e.TraceLoadI16(DATA_LOCAL, i.src1.constant, i.dest);
  }
};
struct LOAD_LOCAL_I32
    : Sequence<LOAD_LOCAL_I32, I<OPCODE_LOAD_LOCAL, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitLocalAccess(e, i.src1.constant(), 4,
                    [&](auto base, uint32_t imm) { e.LDR(i.dest, base, imm); });
    // e.TraceLoadI32(DATA_LOCAL, i.src1.constant, i.dest);
  }
};
struct LOAD_LOCAL_I64
    : Sequence<LOAD_LOCAL_I64, I<OPCODE_LOAD_LOCAL, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitLocalAccess(e, i.src1.constant(), 8,
                    [&](auto base, uint32_t imm) { e.LDR(i.dest, base, imm); });
    // e.TraceLoadI64(DATA_LOCAL, i.src1.constant, i.dest);
  }
};
struct LOAD_LOCAL_F32
    : Sequence<LOAD_LOCAL_F32, I<OPCODE_LOAD_LOCAL, F32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitLocalAccess(e, i.src1.constant(), 4,
                    [&](auto base, uint32_t imm) { e.LDR(i.dest, base, imm); });
    // e.TraceLoadF32(DATA_LOCAL, i.src1.constant, i.dest);
  }
};
struct LOAD_LOCAL_F64
    : Sequence<LOAD_LOCAL_F64, I<OPCODE_LOAD_LOCAL, F64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitLocalAccess(e, i.src1.constant(), 8,
                    [&](auto base, uint32_t imm) { e.LDR(i.dest, base, imm); });
    // e.TraceLoadF64(DATA_LOCAL, i.src1.constant, i.dest);
  }
};
struct LOAD_LOCAL_V128
    : Sequence<LOAD_LOCAL_V128, I<OPCODE_LOAD_LOCAL, V128Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    EmitLocalAccess(e, i.src1.constant(), 16,
                    [&](auto base, uint32_t imm) { e.LDR(i.dest, base, imm); });
    // e.TraceLoadV128(DATA_LOCAL, i.src1.constant, i.dest);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_LOCAL, LOAD_LOCAL_I8, LOAD_LOCAL_I16,
                     LOAD_LOCAL_I32, LOAD_LOCAL_I64, LOAD_LOCAL_F32,
                     LOAD_LOCAL_F64, LOAD_LOCAL_V128);

// ============================================================================
// OPCODE_STORE_LOCAL
// ============================================================================
// Note: all types are always aligned on the stack.
struct STORE_LOCAL_I8
    : Sequence<STORE_LOCAL_I8, I<OPCODE_STORE_LOCAL, VoidOp, I32Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // e.TraceStoreI8(DATA_LOCAL, i.src1.constant, i.src2);
    EmitLocalAccess(e, i.src1.constant(), 1, [&](auto base, uint32_t imm) {
      e.STRB(i.src2, base, imm);
    });
  }
};
struct STORE_LOCAL_I16
    : Sequence<STORE_LOCAL_I16, I<OPCODE_STORE_LOCAL, VoidOp, I32Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // e.TraceStoreI16(DATA_LOCAL, i.src1.constant, i.src2);
    EmitLocalAccess(e, i.src1.constant(), 2, [&](auto base, uint32_t imm) {
      e.STRH(i.src2, base, imm);
    });
  }
};
struct STORE_LOCAL_I32
    : Sequence<STORE_LOCAL_I32, I<OPCODE_STORE_LOCAL, VoidOp, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // e.TraceStoreI32(DATA_LOCAL, i.src1.constant, i.src2);
    EmitLocalAccess(e, i.src1.constant(), 4,
                    [&](auto base, uint32_t imm) { e.STR(i.src2, base, imm); });
  }
};
struct STORE_LOCAL_I64
    : Sequence<STORE_LOCAL_I64, I<OPCODE_STORE_LOCAL, VoidOp, I32Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // e.TraceStoreI64(DATA_LOCAL, i.src1.constant, i.src2);
    EmitLocalAccess(e, i.src1.constant(), 8,
                    [&](auto base, uint32_t imm) { e.STR(i.src2, base, imm); });
  }
};
struct STORE_LOCAL_F32
    : Sequence<STORE_LOCAL_F32, I<OPCODE_STORE_LOCAL, VoidOp, I32Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // e.TraceStoreF32(DATA_LOCAL, i.src1.constant, i.src2);
    EmitLocalAccess(e, i.src1.constant(), 4,
                    [&](auto base, uint32_t imm) { e.STR(i.src2, base, imm); });
  }
};
struct STORE_LOCAL_F64
    : Sequence<STORE_LOCAL_F64, I<OPCODE_STORE_LOCAL, VoidOp, I32Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // e.TraceStoreF64(DATA_LOCAL, i.src1.constant, i.src2);
    EmitLocalAccess(e, i.src1.constant(), 8,
                    [&](auto base, uint32_t imm) { e.STR(i.src2, base, imm); });
  }
};
struct STORE_LOCAL_V128
    : Sequence<STORE_LOCAL_V128, I<OPCODE_STORE_LOCAL, VoidOp, I32Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // e.TraceStoreV128(DATA_LOCAL, i.src1.constant, i.src2);
    EmitLocalAccess(e, i.src1.constant(), 16,
                    [&](auto base, uint32_t imm) { e.STR(i.src2, base, imm); });
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STORE_LOCAL, STORE_LOCAL_I8, STORE_LOCAL_I16,
                     STORE_LOCAL_I32, STORE_LOCAL_I64, STORE_LOCAL_F32,
                     STORE_LOCAL_F64, STORE_LOCAL_V128);

// ============================================================================
// OPCODE_LOAD_CONTEXT
// ============================================================================
struct LOAD_CONTEXT_I8
    : Sequence<LOAD_CONTEXT_I8, I<OPCODE_LOAD_CONTEXT, I8Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.LDRB(i.dest, e.GetContextReg(), i.src1.value);
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.LDRB(e.GetNativeParam(1).toW(), e.GetContextReg(), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextLoadI8));
    }
  }
};
struct LOAD_CONTEXT_I16
    : Sequence<LOAD_CONTEXT_I16, I<OPCODE_LOAD_CONTEXT, I16Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.LDRH(i.dest, e.GetContextReg(), i.src1.value);
    if (IsTracingData()) {
      e.LDRH(e.GetNativeParam(1).toW(), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextLoadI16));
    }
  }
};
struct LOAD_CONTEXT_I32
    : Sequence<LOAD_CONTEXT_I32, I<OPCODE_LOAD_CONTEXT, I32Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.LDR(i.dest, e.GetContextReg(), i.src1.value);
    if (IsTracingData()) {
      e.LDR(e.GetNativeParam(1).toW(), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextLoadI32));
    }
  }
};
struct LOAD_CONTEXT_I64
    : Sequence<LOAD_CONTEXT_I64, I<OPCODE_LOAD_CONTEXT, I64Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.LDR(i.dest, e.GetContextReg(), i.src1.value);
    if (IsTracingData()) {
      e.LDR(e.GetNativeParam(1), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextLoadI64));
    }
  }
};
struct LOAD_CONTEXT_F32
    : Sequence<LOAD_CONTEXT_F32, I<OPCODE_LOAD_CONTEXT, F32Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.LDR(i.dest, e.GetContextReg(), i.src1.value);
    if (IsTracingData()) {
      e.ADD(e.GetNativeParam(1), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextLoadF32));
    }
  }
};
struct LOAD_CONTEXT_F64
    : Sequence<LOAD_CONTEXT_F64, I<OPCODE_LOAD_CONTEXT, F64Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.LDR(i.dest, e.GetContextReg(), i.src1.value);
    if (IsTracingData()) {
      e.ADD(e.GetNativeParam(1), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextLoadF64));
    }
  }
};
struct LOAD_CONTEXT_V128
    : Sequence<LOAD_CONTEXT_V128, I<OPCODE_LOAD_CONTEXT, V128Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.LDR(i.dest, e.GetContextReg(), i.src1.value);
    if (IsTracingData()) {
      e.ADD(e.GetNativeParam(1), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextLoadV128));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_CONTEXT, LOAD_CONTEXT_I8, LOAD_CONTEXT_I16,
                     LOAD_CONTEXT_I32, LOAD_CONTEXT_I64, LOAD_CONTEXT_F32,
                     LOAD_CONTEXT_F64, LOAD_CONTEXT_V128);

// ============================================================================
// OPCODE_STORE_CONTEXT
// ============================================================================
// Note: all types are always aligned on the stack.
struct STORE_CONTEXT_I8
    : Sequence<STORE_CONTEXT_I8,
               I<OPCODE_STORE_CONTEXT, VoidOp, OffsetOp, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src2.is_constant) {
      e.MOV(W0, i.src2.constant());
      e.STRB(W0, e.GetContextReg(), i.src1.value);
    } else {
      e.STRB(i.src2.reg(), e.GetContextReg(), i.src1.value);
    }
    if (IsTracingData()) {
      e.LDRB(e.GetNativeParam(1).toW(), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextStoreI8));
    }
  }
};
struct STORE_CONTEXT_I16
    : Sequence<STORE_CONTEXT_I16,
               I<OPCODE_STORE_CONTEXT, VoidOp, OffsetOp, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src2.is_constant) {
      e.MOV(W0, i.src2.constant());
      e.STRH(W0, e.GetContextReg(), i.src1.value);
    } else {
      e.STRH(i.src2.reg(), e.GetContextReg(), i.src1.value);
    }
    if (IsTracingData()) {
      e.LDRH(e.GetNativeParam(1).toW(), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextStoreI16));
    }
  }
};
struct STORE_CONTEXT_I32
    : Sequence<STORE_CONTEXT_I32,
               I<OPCODE_STORE_CONTEXT, VoidOp, OffsetOp, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src2.is_constant) {
      e.MOV(W0, i.src2.constant());
      e.STR(W0, e.GetContextReg(), i.src1.value);
    } else {
      e.STR(i.src2.reg(), e.GetContextReg(), i.src1.value);
    }
    if (IsTracingData()) {
      e.LDR(e.GetNativeParam(1).toW(), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextStoreI32));
    }
  }
};
struct STORE_CONTEXT_I64
    : Sequence<STORE_CONTEXT_I64,
               I<OPCODE_STORE_CONTEXT, VoidOp, OffsetOp, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src2.is_constant) {
      e.MOV(X0, i.src2.constant());
      e.STR(X0, e.GetContextReg(), i.src1.value);
    } else {
      e.STR(i.src2.reg(), e.GetContextReg(), i.src1.value);
    }
    if (IsTracingData()) {
      e.LDR(e.GetNativeParam(1), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextStoreI64));
    }
  }
};
struct STORE_CONTEXT_F32
    : Sequence<STORE_CONTEXT_F32,
               I<OPCODE_STORE_CONTEXT, VoidOp, OffsetOp, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src2.is_constant) {
      e.MOV(W0, i.src2.value->constant.i32);
      e.STR(W0, e.GetContextReg(), i.src1.value);
    } else {
      e.STR(i.src2, e.GetContextReg(), i.src1.value);
    }
    if (IsTracingData()) {
      e.ADD(e.GetNativeParam(1), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextStoreF32));
    }
  }
};
struct STORE_CONTEXT_F64
    : Sequence<STORE_CONTEXT_F64,
               I<OPCODE_STORE_CONTEXT, VoidOp, OffsetOp, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src2.is_constant) {
      e.MOV(X0, i.src2.value->constant.i64);
      e.STR(X0, e.GetContextReg(), i.src1.value);
    } else {
      e.STR(i.src2, e.GetContextReg(), i.src1.value);
    }
    if (IsTracingData()) {
      e.ADD(e.GetNativeParam(1), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextStoreF64));
    }
  }
};
struct STORE_CONTEXT_V128
    : Sequence<STORE_CONTEXT_V128,
               I<OPCODE_STORE_CONTEXT, VoidOp, OffsetOp, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.src2.is_constant) {
      e.LoadConstantV(Q0, i.src2.constant());
      e.STR(Q0, e.GetContextReg(), i.src1.value);
    } else {
      e.STR(i.src2, e.GetContextReg(), i.src1.value);
    }
    if (IsTracingData()) {
      e.ADD(e.GetNativeParam(1), e.GetContextReg(), i.src1.value);
      e.MOV(e.GetNativeParam(0), i.src1.value);
      e.CallNative(reinterpret_cast<void*>(TraceContextStoreV128));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STORE_CONTEXT, STORE_CONTEXT_I8, STORE_CONTEXT_I16,
                     STORE_CONTEXT_I32, STORE_CONTEXT_I64, STORE_CONTEXT_F32,
                     STORE_CONTEXT_F64, STORE_CONTEXT_V128);

// ============================================================================
// OPCODE_LOAD_MMIO
// ============================================================================
// Note: all types are always aligned in the context.
struct LOAD_MMIO_I32
    : Sequence<LOAD_MMIO_I32, I<OPCODE_LOAD_MMIO, I32Op, OffsetOp, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // uint64_t (context, addr)
    const auto mmio_range = reinterpret_cast<MMIORange*>(i.src1.value);
    const auto read_address = uint32_t(i.src2.value);
    e.MOV(e.GetNativeParam(0), uint64_t(mmio_range->callback_context));
    e.MOV(e.GetNativeParam(1).toW(), read_address);
    e.CallNativeSafe(reinterpret_cast<void*>(mmio_range->read));
    e.REV(i.dest, W0);
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(0).toW(), i.dest);
      e.MOV(X1, read_address);
      e.CallNative(reinterpret_cast<void*>(TraceContextLoadI32));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_MMIO, LOAD_MMIO_I32);

// ============================================================================
// OPCODE_STORE_MMIO
// ============================================================================
// Note: all types are always aligned on the stack.
struct STORE_MMIO_I32
    : Sequence<STORE_MMIO_I32,
               I<OPCODE_STORE_MMIO, VoidOp, OffsetOp, OffsetOp, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // void (context, addr, value)
    const auto mmio_range = reinterpret_cast<MMIORange*>(i.src1.value);
    const auto write_address = uint32_t(i.src2.value);
    e.MOV(e.GetNativeParam(0), uint64_t(mmio_range->callback_context));
    e.MOV(e.GetNativeParam(1).toW(), write_address);
    if (i.src3.is_constant) {
      e.MOV(e.GetNativeParam(2).toW(), xe::byte_swap(i.src3.constant()));
    } else {
      e.REV(e.GetNativeParam(2).toW(), i.src3);
    }
    e.CallNativeSafe(reinterpret_cast<void*>(mmio_range->write));
    if (IsTracingData()) {
      if (i.src3.is_constant) {
        e.MOV(e.GetNativeParam(0).toW(), i.src3.constant());
      } else {
        e.MOV(e.GetNativeParam(0).toW(), i.src3);
      }
      e.MOV(X1, write_address);
      e.CallNative(reinterpret_cast<void*>(TraceContextStoreI32));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STORE_MMIO, STORE_MMIO_I32);

// ============================================================================
// OPCODE_LOAD_OFFSET
// ============================================================================
struct LOAD_OFFSET_I8
    : Sequence<LOAD_OFFSET_I8, I<OPCODE_LOAD_OFFSET, I8Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddressOffset(e, i.src1, i.src2);
    e.LDRB(i.dest, addr_reg);
  }
};

struct LOAD_OFFSET_I16
    : Sequence<LOAD_OFFSET_I16, I<OPCODE_LOAD_OFFSET, I16Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddressOffset(e, i.src1, i.src2);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.LDRH(i.dest, addr_reg);
      e.REV16(i.dest, i.dest);
    } else {
      e.LDRH(i.dest, addr_reg);
    }
  }
};

struct LOAD_OFFSET_I32
    : Sequence<LOAD_OFFSET_I32, I<OPCODE_LOAD_OFFSET, I32Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const bool force_mmio_byteswap =
        cvars::a64_force_mmio_aware_byteswap_loads &&
        (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP);
    if (force_mmio_byteswap || IsPossibleMMIOInstruction(e, i.instr)) {
      void* addrptr = (void*)&MMIOAwareLoad<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        addrptr = (void*)&MMIOAwareLoad<uint32_t, true>;
      }
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto offset_reg = W3;
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(offset_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(offset_reg, i.src2.reg().toW());
      }
      e.ADD(guest_addr_reg, guest_addr_reg, offset_reg);
      e.CallNativeSafe(addrptr);
      e.MOV(i.dest, W0);
      return;
    }
    auto addr_reg = ComputeMemoryAddressOffset(e, i.src1, i.src2);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.LDR(i.dest, addr_reg);
      e.REV(i.dest, i.dest);
    } else {
      e.LDR(i.dest, addr_reg);
    }
  }
};

struct LOAD_OFFSET_I64
    : Sequence<LOAD_OFFSET_I64, I<OPCODE_LOAD_OFFSET, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddressOffset(e, i.src1, i.src2);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.LDR(i.dest, addr_reg);
      e.REV(i.dest, i.dest);
    } else {
      e.LDR(i.dest, addr_reg);
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_OFFSET, LOAD_OFFSET_I8, LOAD_OFFSET_I16,
                     LOAD_OFFSET_I32, LOAD_OFFSET_I64);

// ============================================================================
// OPCODE_STORE_OFFSET
// ============================================================================
struct STORE_OFFSET_I8
    : Sequence<STORE_OFFSET_I8,
               I<OPCODE_STORE_OFFSET, VoidOp, I64Op, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddressOffset(e, i.src1, i.src2);
    if (i.src3.is_constant) {
      e.MOV(W0, i.src3.constant());
      e.STRB(W0, addr_reg);
    } else {
      e.STRB(i.src3, addr_reg);
    }
  }
};

struct STORE_OFFSET_I16
    : Sequence<STORE_OFFSET_I16,
               I<OPCODE_STORE_OFFSET, VoidOp, I64Op, I64Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      void* addrptr = (void*)&MMIOAwareStore<uint16_t, true>;
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toW();
      const auto offset_reg = W3;
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(offset_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(offset_reg, i.src2.reg().toW());
      }
      e.ADD(guest_addr_reg, guest_addr_reg, offset_reg);
      if (i.src3.is_constant) {
        e.MOV(value_reg, uint32_t(i.src3.constant()));
      } else {
        e.MOV(value_reg, i.src3.reg().toW());
      }
      e.CallNativeSafe(addrptr);
      return;
    } else {
      auto addr_reg = ComputeMemoryAddressOffset(e, i.src1, i.src2);
      if (i.src3.is_constant) {
        e.MOV(W0, i.src3.constant());
        e.STRH(W0, addr_reg);
      } else {
        e.STRH(i.src3, addr_reg);
      }
    }
  }
};

struct STORE_OFFSET_I32
    : Sequence<STORE_OFFSET_I32,
               I<OPCODE_STORE_OFFSET, VoidOp, I64Op, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const uint32_t watch_addr = cvars::a64_watch_store_address;
    if (watch_addr) {
      oaknut::Label skip_watch;
      if (i.src1.is_constant) {
        e.MOV(W0, uint32_t(i.src1.constant()));
      } else {
        e.MOV(W0, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(W2, uint32_t(i.src2.constant()));
        e.ADD(W0, W0, W2);
      } else {
        e.ADD(W0, W0, i.src2.reg().toW());
      }
      e.MOV(W1, watch_addr);
      e.CMP(W0, W1);
      e.B(Cond::NE, skip_watch);
      e.MOV(e.GetNativeParam(0).toW(), W0);
      if (i.src3.is_constant) {
        e.MOV(e.GetNativeParam(1).toW(), uint32_t(i.src3.constant()));
      } else {
        e.MOV(e.GetNativeParam(1).toW(), i.src3);
      }
      e.MOV(e.GetNativeParam(2).toW(), i.instr->GuestAddressFor());
      e.CallNativeSafe(reinterpret_cast<void*>(LogStoreWatch32));
      e.l(skip_watch);
    }

    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      void* addrptr = (void*)&MMIOAwareStore<uint32_t, true>;
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toW();
      const auto offset_reg = W3;
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(offset_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(offset_reg, i.src2.reg().toW());
      }
      e.ADD(guest_addr_reg, guest_addr_reg, offset_reg);
      if (i.src3.is_constant) {
        e.MOV(value_reg, uint32_t(i.src3.constant()));
      } else {
        e.MOV(value_reg, i.src3);
      }
      e.CallNativeSafe(addrptr);
      return;
    }
    if (IsPossibleMMIOInstruction(e, i.instr)) {
      void* addrptr = (void*)&MMIOAwareStore<uint32_t, false>;
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toW();
      const auto offset_reg = W3;
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(offset_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(offset_reg, i.src2.reg().toW());
      }
      e.ADD(guest_addr_reg, guest_addr_reg, offset_reg);
      if (i.src3.is_constant) {
        e.MOV(value_reg, uint32_t(i.src3.constant()));
      } else {
        e.MOV(value_reg, i.src3);
      }
      e.CallNativeSafe(addrptr);
      return;
    }
    auto addr_reg = ComputeMemoryAddressOffset(e, i.src1, i.src2);
    if (i.src3.is_constant) {
      e.MOV(W0, i.src3.constant());
      e.STR(W0, addr_reg);
    } else {
      e.STR(i.src3, addr_reg);
    }
  }
};

struct STORE_OFFSET_I64
    : Sequence<STORE_OFFSET_I64,
               I<OPCODE_STORE_OFFSET, VoidOp, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const uint32_t watch_addr = cvars::a64_watch_store_address;
    if (watch_addr) {
      oaknut::Label skip_watch;
      oaknut::Label do_watch;
      if (i.src1.is_constant) {
        e.MOV(W0, uint32_t(i.src1.constant()));
      } else {
        e.MOV(W0, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(W2, uint32_t(i.src2.constant()));
        e.ADD(W0, W0, W2);
      } else {
        e.ADD(W0, W0, i.src2.reg().toW());
      }
      e.MOV(W1, watch_addr);
      e.CMP(W0, W1);
      e.B(Cond::EQ, do_watch);
      e.ADD(W2, W0, 4);
      e.CMP(W2, W1);
      e.B(Cond::NE, skip_watch);
      e.l(do_watch);
      e.MOV(e.GetNativeParam(0).toW(), W0);
      if (i.src3.is_constant) {
        e.MOV(e.GetNativeParam(1).toX(), i.src3.constant());
      } else {
        e.MOV(e.GetNativeParam(1).toX(), i.src3);
      }
      e.MOV(e.GetNativeParam(2).toW(), i.instr->GuestAddressFor());
      e.CallNativeSafe(reinterpret_cast<void*>(LogStoreWatch64));
      e.l(skip_watch);
    }
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      void* addrptr = (void*)&MMIOAwareStore<uint64_t, true>;
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toX();
      const auto offset_reg = W3;
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(offset_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(offset_reg, i.src2.reg().toW());
      }
      e.ADD(guest_addr_reg, guest_addr_reg, offset_reg);
      if (i.src3.is_constant) {
        e.MOV(value_reg, i.src3.constant());
      } else {
        e.MOV(value_reg, i.src3);
      }
      e.CallNativeSafe(addrptr);
      return;
    } else {
      auto addr_reg = ComputeMemoryAddressOffset(e, i.src1, i.src2);
      if (i.src3.is_constant) {
        e.MovMem64(addr_reg, 0, i.src3.constant());
      } else {
        e.STR(i.src3, addr_reg);
      }
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STORE_OFFSET, STORE_OFFSET_I8, STORE_OFFSET_I16,
                     STORE_OFFSET_I32, STORE_OFFSET_I64);

// ============================================================================
// OPCODE_LOAD
// ============================================================================
struct LOAD_I8 : Sequence<LOAD_I8, I<OPCODE_LOAD, I8Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    e.LDRB(i.dest, addr_reg);
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(1).toW(), i.dest);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryLoadI8));
    }
  }
};
struct LOAD_I16 : Sequence<LOAD_I16, I<OPCODE_LOAD, I16Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.LDRH(i.dest, addr_reg);
      e.REV16(i.dest, i.dest);
    } else {
      e.LDRH(i.dest, addr_reg);
    }
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(1).toW(), i.dest);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryLoadI16));
    }
  }
};
struct LOAD_I32 : Sequence<LOAD_I32, I<OPCODE_LOAD, I32Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const bool force_mmio_byteswap =
        cvars::a64_force_mmio_aware_byteswap_loads &&
        (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP);
    if (force_mmio_byteswap || IsPossibleMMIOInstruction(e, i.instr)) {
      void* addrptr = (void*)&MMIOAwareLoad<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        addrptr = (void*)&MMIOAwareLoad<uint32_t, true>;
      }
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      e.CallNativeSafe(addrptr);
      e.MOV(i.dest, W0);
      return;
    }
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.LDR(i.dest, addr_reg);
      e.REV(i.dest, i.dest);
    } else {
      e.LDR(i.dest, addr_reg);
    }
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(1).toW(), i.dest);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryLoadI32));
    }
  }
};
struct LOAD_I64 : Sequence<LOAD_I64, I<OPCODE_LOAD, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.LDR(i.dest, addr_reg);
      e.REV64(i.dest, i.dest);
    } else {
      e.LDR(i.dest, addr_reg);
    }
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(1), i.dest);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryLoadI64));
    }
  }
};
struct LOAD_F32 : Sequence<LOAD_F32, I<OPCODE_LOAD, F32Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    e.LDR(i.dest, addr_reg);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      assert_always("not implemented yet");
    }
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(1), addr_reg);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryLoadF32));
    }
  }
};
struct LOAD_F64 : Sequence<LOAD_F64, I<OPCODE_LOAD, F64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    e.LDR(i.dest, addr_reg);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      assert_always("not implemented yet");
    }
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(1), addr_reg);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryLoadF64));
    }
  }
};
struct LOAD_V128 : Sequence<LOAD_V128, I<OPCODE_LOAD, V128Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    e.LDR(i.dest, addr_reg);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      // Reverse upper and lower 64-bit halfs
      e.REV64(i.dest.reg().B16(), i.dest.reg().B16());
      // Reverse the 64-bit halfs themselves
      e.EXT(i.dest.reg().B16(), i.dest.reg().B16(), i.dest.reg().B16(), 8);
    }
    if (IsTracingData()) {
      e.MOV(e.GetNativeParam(1), addr_reg);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryLoadV128));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD, LOAD_I8, LOAD_I16, LOAD_I32, LOAD_I64,
                     LOAD_F32, LOAD_F64, LOAD_V128);

// ============================================================================
// OPCODE_STORE
// ============================================================================
// Note: most *should* be aligned, but needs to be checked!
struct STORE_I8 : Sequence<STORE_I8, I<OPCODE_STORE, VoidOp, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    if (i.src2.is_constant) {
      e.MOV(W0, i.src2.constant());
      e.STRB(W0, addr_reg);
    } else {
      e.STRB(i.src2.reg(), addr_reg);
    }
    if (IsTracingData()) {
      addr_reg = ComputeMemoryAddress(e, i.src1);
      e.LDRB(e.GetNativeParam(1).toW(), addr_reg);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryStoreI8));
    }
  }
};
struct STORE_I16 : Sequence<STORE_I16, I<OPCODE_STORE, VoidOp, I64Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      void* addrptr = (void*)&MMIOAwareStore<uint16_t, true>;
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toW();
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(value_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(value_reg, i.src2.reg().toW());
      }
      e.CallNativeSafe(addrptr);
      return;
    } else {
      auto addr_reg = ComputeMemoryAddress(e, i.src1);
      if (i.src2.is_constant) {
        e.MOV(W0, i.src2.constant());
        e.STRH(W0, addr_reg);
      } else {
        e.STRH(i.src2.reg(), addr_reg);
      }
    }
    if (IsTracingData()) {
      auto trace_addr_reg = ComputeMemoryAddress(e, i.src1);
      e.LDRH(e.GetNativeParam(1).toW(), trace_addr_reg);
      e.MOV(e.GetNativeParam(0), trace_addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryStoreI16));
    }
  }
};
struct STORE_I32 : Sequence<STORE_I32, I<OPCODE_STORE, VoidOp, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const uint32_t watch_addr = cvars::a64_watch_store_address;
    if (watch_addr) {
      oaknut::Label skip_watch;
      if (i.src1.is_constant) {
        e.MOV(W0, uint32_t(i.src1.constant()));
      } else {
        e.MOV(W0, i.src1.reg().toW());
      }
      e.MOV(W1, watch_addr);
      e.CMP(W0, W1);
      e.B(Cond::NE, skip_watch);
      e.MOV(e.GetNativeParam(0).toW(), W0);
      if (i.src2.is_constant) {
        e.MOV(e.GetNativeParam(1).toW(), uint32_t(i.src2.constant()));
      } else {
        e.MOV(e.GetNativeParam(1).toW(), i.src2);
      }
      e.MOV(e.GetNativeParam(2).toW(), i.instr->GuestAddressFor());
      e.CallNativeSafe(reinterpret_cast<void*>(LogStoreWatch32));
      e.l(skip_watch);
    }

    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      void* addrptr = (void*)&MMIOAwareStore<uint32_t, true>;
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toW();
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(value_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(value_reg, i.src2.reg());
      }
      e.CallNativeSafe(addrptr);
      return;
    }
    if (IsPossibleMMIOInstruction(e, i.instr)) {
      void* addrptr = (void*)&MMIOAwareStore<uint32_t, false>;
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toW();
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(value_reg, uint32_t(i.src2.constant()));
      } else {
        e.MOV(value_reg, i.src2.reg());
      }
      e.CallNativeSafe(addrptr);
      return;
    }
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    if (i.src2.is_constant) {
      e.MOV(W0, i.src2.constant());
      e.STR(W0, addr_reg);
    } else {
      e.STR(i.src2.reg(), addr_reg);
    }
    if (IsTracingData()) {
      addr_reg = ComputeMemoryAddress(e, i.src1);
      e.LDR(e.GetNativeParam(1).toW(), addr_reg);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryStoreI32));
    }
  }
};
struct STORE_I64 : Sequence<STORE_I64, I<OPCODE_STORE, VoidOp, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const uint32_t watch_addr = cvars::a64_watch_store_address;
    if (watch_addr) {
      oaknut::Label skip_watch;
      oaknut::Label do_watch;
      if (i.src1.is_constant) {
        e.MOV(W0, uint32_t(i.src1.constant()));
      } else {
        e.MOV(W0, i.src1.reg().toW());
      }
      e.MOV(W1, watch_addr);
      e.CMP(W0, W1);
      e.B(Cond::EQ, do_watch);
      e.ADD(W2, W0, 4);
      e.CMP(W2, W1);
      e.B(Cond::NE, skip_watch);
      e.l(do_watch);
      e.MOV(e.GetNativeParam(0).toW(), W0);
      if (i.src2.is_constant) {
        e.MOV(e.GetNativeParam(1).toX(), i.src2.constant());
      } else {
        e.MOV(e.GetNativeParam(1).toX(), i.src2);
      }
      e.MOV(e.GetNativeParam(2).toW(), i.instr->GuestAddressFor());
      e.CallNativeSafe(reinterpret_cast<void*>(LogStoreWatch64));
      e.l(skip_watch);
    }
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      void* addrptr = (void*)&MMIOAwareStore<uint64_t, true>;
      const auto guest_addr_reg = e.GetNativeParam(0).toW();
      const auto value_reg = e.GetNativeParam(1).toX();
      if (i.src1.is_constant) {
        e.MOV(guest_addr_reg, uint32_t(i.src1.constant()));
      } else {
        e.MOV(guest_addr_reg, i.src1.reg().toW());
      }
      if (i.src2.is_constant) {
        e.MOV(value_reg, i.src2.constant());
      } else {
        e.MOV(value_reg, i.src2.reg());
      }
      e.CallNativeSafe(addrptr);
      return;
    } else {
      auto addr_reg = ComputeMemoryAddress(e, i.src1);
      if (i.src2.is_constant) {
        e.MovMem64(addr_reg, 0, i.src2.constant());
      } else {
        e.STR(i.src2.reg(), addr_reg);
      }
    }
    if (IsTracingData()) {
      auto trace_addr_reg = ComputeMemoryAddress(e, i.src1);
      e.LDR(e.GetNativeParam(1), trace_addr_reg);
      e.MOV(e.GetNativeParam(0), trace_addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryStoreI64));
    }
  }
};
struct STORE_F32 : Sequence<STORE_F32, I<OPCODE_STORE, VoidOp, I64Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      assert_false(i.src2.is_constant);
      assert_always("not yet implemented");
    } else {
      if (i.src2.is_constant) {
        e.MOV(W0, i.src2.value->constant.i32);
        e.STR(W0, addr_reg);
      } else {
        e.STR(i.src2, addr_reg);
      }
    }
    if (IsTracingData()) {
      addr_reg = ComputeMemoryAddress(e, i.src1);
      e.MOV(e.GetNativeParam(1), addr_reg);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryStoreF32));
    }
  }
};
struct STORE_F64 : Sequence<STORE_F64, I<OPCODE_STORE, VoidOp, I64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      assert_false(i.src2.is_constant);
      assert_always("not yet implemented");
    } else {
      if (i.src2.is_constant) {
        e.MOV(X0, i.src2.value->constant.i64);
        e.STR(X0, addr_reg);
      } else {
        e.STR(i.src2, addr_reg);
      }
    }
    if (IsTracingData()) {
      addr_reg = ComputeMemoryAddress(e, i.src1);
      e.MOV(e.GetNativeParam(1), addr_reg);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryStoreF64));
    }
  }
};
struct STORE_V128
    : Sequence<STORE_V128, I<OPCODE_STORE, VoidOp, I64Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      assert_false(i.src2.is_constant);
      // Reverse upper and lower 64-bit halfs
      e.REV64(Q0.B16(), i.src2.reg().B16());
      // Reverse the 64-bit halfs themselves
      e.EXT(Q0.B16(), Q0.B16(), Q0.B16(), 8);
      e.STR(Q0, addr_reg);
    } else {
      if (i.src2.is_constant) {
        e.LoadConstantV(Q0, i.src2.constant());
        e.STR(Q0, addr_reg);
      } else {
        e.STR(i.src2, addr_reg);
      }
    }
    if (IsTracingData()) {
      addr_reg = ComputeMemoryAddress(e, i.src1);
      e.MOV(e.GetNativeParam(1), addr_reg);
      e.MOV(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemoryStoreV128));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STORE, STORE_I8, STORE_I16, STORE_I32, STORE_I64,
                     STORE_F32, STORE_F64, STORE_V128);

// ============================================================================
// OPCODE_CACHE_CONTROL
// ============================================================================
struct CACHE_CONTROL
    : Sequence<CACHE_CONTROL,
               I<OPCODE_CACHE_CONTROL, VoidOp, I64Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    bool is_clflush = false, is_prefetch = false;
    switch (CacheControlType(i.instr->flags)) {
      case CacheControlType::CACHE_CONTROL_TYPE_DATA_TOUCH:
      case CacheControlType::CACHE_CONTROL_TYPE_DATA_TOUCH_FOR_STORE:
        is_prefetch = true;
        break;
      case CacheControlType::CACHE_CONTROL_TYPE_DATA_STORE:
      case CacheControlType::CACHE_CONTROL_TYPE_DATA_STORE_AND_FLUSH:
        is_clflush = true;
        break;
      default:
        assert_unhandled_case(CacheControlType(i.instr->flags));
        return;
    }
    size_t cache_line_size = i.src2.value;

    XReg addr = X0;
    uint32_t address_constant;
    if (i.src1.is_constant) {
      // TODO(benvanik): figure out how to do this without a temp.
      // Since the constant is often 0x8... if we tried to use that as a
      // displacement it would be sign extended and mess things up.
      address_constant = static_cast<uint32_t>(i.src1.constant());
      if (address_constant < 0x80000000) {
        e.ADD(addr, e.GetMembaseReg(), address_constant);
      } else {
        if (address_constant >= 0xE0000000 &&
            xe::memory::allocation_granularity() > 0x1000) {
          e.MOV(X1, address_constant + 0x1000);
        } else {
          e.MOV(X1, address_constant);
        }
        e.ADD(addr, e.GetMembaseReg(), X1);
      }
    } else {
      if (xe::memory::allocation_granularity() > 0x1000) {
        // Emulate the 4 KB physical address offset in 0xE0000000+ when can't do
        // it via memory mapping.
        e.MOV(X1, 0xE0000000);
        e.CMP(i.src1.reg(), X1);
        e.CSET(X1, Cond::HS);
        e.ADD(X1, i.src1.reg(), X1, LSL, 12);
      } else {
        // Clear the top 32 bits, as they are likely garbage.
        e.MOV(W1, i.src1.reg().toW());
      }
      e.ADD(addr, e.GetMembaseReg(), X1);
    }

    if (is_clflush) {
      // TODO(wunkolo): These kind of cache-maintenance instructions cause an
      // illegal-instruction on windows, but is trapped to proper EL1 code on
      // Linux. Need a way to do cache-maintenance on Windows-Arm
      // e.DC(DcOp::CIVAC, addr);

      // Full data sync
      e.DSB(BarrierOp::ISH);
    }
    if (is_prefetch) {
      e.PRFM(PrfOp::PLDL1KEEP, addr);
    }

    if (cache_line_size >= 128) {
      // Prefetch the other 64 bytes of the 128-byte cache line.
      if (i.src1.is_constant && address_constant < 0x80000000) {
        e.ADD(addr, e.GetMembaseReg(), address_constant ^ 64);
      } else {
        e.EOR(X1, X1, 64);
      }
      if (is_clflush) {
        // TODO(wunkolo): These kind of cache-maintenance instructions cause an
        // illegal-instruction on windows, but is trapped to proper EL1 code on
        // Linux. Need a way to do cache-maintenance on Windows-Arm
        // e.DC(DcOp::CIVAC, addr);

        // Full data sync
        e.DSB(BarrierOp::ISH);
      }
      if (is_prefetch) {
        e.PRFM(PrfOp::PLDL1KEEP, addr);
      }
      assert_true(cache_line_size == 128);
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CACHE_CONTROL, CACHE_CONTROL);

// ============================================================================
// OPCODE_MEMORY_BARRIER
// ============================================================================
struct MEMORY_BARRIER
    : Sequence<MEMORY_BARRIER, I<OPCODE_MEMORY_BARRIER, VoidOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.DMB(BarrierOp::SY);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MEMORY_BARRIER, MEMORY_BARRIER);

// ============================================================================
// OPCODE_MEMSET
// ============================================================================
struct MEMSET_I64_I8_I64
    : Sequence<MEMSET_I64_I8_I64,
               I<OPCODE_MEMSET, VoidOp, I64Op, I8Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(i.src2.is_constant);
    assert_true(i.src3.is_constant);
    assert_true(i.src2.constant() == 0);
    e.MOVI(Q0.B16(), 0);
    auto addr_reg = ComputeMemoryAddress(e, i.src1);
    switch (i.src3.constant()) {
      case 32:
        e.STP(Q0, Q0, addr_reg, 0 * 16);
        break;
      case 128:
        e.STP(Q0, Q0, addr_reg, 0 * 16);
        e.STP(Q0, Q0, addr_reg, 2 * 16);
        e.STP(Q0, Q0, addr_reg, 4 * 16);
        e.STP(Q0, Q0, addr_reg, 6 * 16);
        break;
      default:
        assert_unhandled_case(i.src3.constant());
        break;
    }
    if (IsTracingData()) {
      addr_reg = ComputeMemoryAddress(e, i.src1);
      e.MOV(e.GetNativeParam(2), i.src3.constant());
      e.MOV(e.GetNativeParam(1), i.src2.constant());
      e.LDR(e.GetNativeParam(0), addr_reg);
      e.CallNative(reinterpret_cast<void*>(TraceMemset));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MEMSET, MEMSET_I64_I8_I64);

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
