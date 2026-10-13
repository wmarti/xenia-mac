/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Randomized differential tests for the VMX operations that dominate Bink
// video decoding (vaddshs, vsubshs, vsrah, vslh, vmrghb/vmrglb, vsldoi,
// vpkshus). Each backend result is checked against a scalar reference written
// in PowerPC element order, so the tests are independent of the host register
// layout.

#include <cstdint>
#include <random>

#include "xenia/cpu/testing/util.h"

using namespace xe;
using namespace xe::cpu;
using namespace xe::cpu::hir;
using namespace xe::cpu::testing;
using xe::cpu::ppc::PPCContext;

namespace {

// Xenia keeps each 32-bit word of a guest vector in host byte order.
uint8_t& B(vec128_t& v, int j) { return v.u8[j ^ 3]; }
uint16_t& H(vec128_t& v, int j) { return v.u16[j ^ 1]; }
int16_t SH(vec128_t& v, int j) { return int16_t(H(v, j)); }

vec128_t RandomVec(std::mt19937& rng) {
  vec128_t v;
  for (int k = 0; k < 4; ++k) v.u32[k] = rng();
  return v;
}

// Biased toward the small signed values an IDCT produces.
vec128_t RandomSmallH(std::mt19937& rng) {
  vec128_t v;
  for (int j = 0; j < 8; ++j) {
    H(v, j) = uint16_t(int16_t(int(rng() % 1024) - 512));
  }
  return v;
}

int16_t SatS16(int x) {
  return int16_t(x < -32768 ? -32768 : (x > 32767 ? 32767 : x));
}
uint8_t SatU8(int x) { return uint8_t(x < 0 ? 0 : (x > 255 ? 255 : x)); }

constexpr int kIters = 200;

}  // namespace

TEST_CASE("VMX_FUZZ_VADDSHS_VSUBSHS", "[instr][vmxfuzz]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 3, b.VectorAdd(LoadVR(b, 4), LoadVR(b, 5), INT16_TYPE,
                              ARITHMETIC_SATURATE));
    StoreVR(b, 6, b.VectorSub(LoadVR(b, 4), LoadVR(b, 5), INT16_TYPE,
                              ARITHMETIC_SATURATE));
    b.Return();
  });
  std::mt19937 rng(1);
  for (int it = 0; it < kIters; ++it) {
    vec128_t a = (it & 1) ? RandomVec(rng) : RandomSmallH(rng);
    vec128_t c = (it & 2) ? RandomVec(rng) : RandomSmallH(rng);
    test.Run([&](PPCContext* ctx) { ctx->v[4] = a; ctx->v[5] = c; },
             [&](PPCContext* ctx) {
               for (int j = 0; j < 8; ++j) {
                 INFO("it " << it << " lane " << j);
                 REQUIRE(SH(ctx->v[3], j) == SatS16(SH(a, j) + SH(c, j)));
                 REQUIRE(SH(ctx->v[6], j) == SatS16(SH(a, j) - SH(c, j)));
               }
             });
  }
}

TEST_CASE("VMX_FUZZ_VSRAH_VSLH_REG", "[instr][vmxfuzz]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 3, b.VectorSha(LoadVR(b, 4), LoadVR(b, 5), INT16_TYPE));
    StoreVR(b, 6, b.VectorShl(LoadVR(b, 4), LoadVR(b, 5), INT16_TYPE));
    b.Return();
  });
  std::mt19937 rng(2);
  for (int it = 0; it < kIters; ++it) {
    vec128_t a = RandomVec(rng);
    vec128_t s = RandomVec(rng);
    if (it & 1) {  // splatted count, as from vspltish
      uint16_t c = uint16_t(rng());
      for (int j = 0; j < 8; ++j) H(s, j) = c;
    }
    test.Run([&](PPCContext* ctx) { ctx->v[4] = a; ctx->v[5] = s; },
             [&](PPCContext* ctx) {
               for (int j = 0; j < 8; ++j) {
                 INFO("it " << it << " lane " << j);
                 int sh = H(s, j) & 15;
                 REQUIRE(SH(ctx->v[3], j) == int16_t(SH(a, j) >> sh));
                 REQUIRE(H(ctx->v[6], j) == uint16_t(H(a, j) << sh));
               }
             });
  }
}

TEST_CASE("VMX_FUZZ_VSRAH_VSLH_CONST", "[instr][vmxfuzz]") {
  for (int sh = 0; sh < 16; ++sh) {
    TestFunction test([sh](HIRBuilder& b) {
      vec128_t s;
      for (int j = 0; j < 8; ++j) H(s, j) = uint16_t(sh);
      StoreVR(b, 3,
              b.VectorSha(LoadVR(b, 4), b.LoadConstantVec128(s), INT16_TYPE));
      StoreVR(b, 6,
              b.VectorShl(LoadVR(b, 4), b.LoadConstantVec128(s), INT16_TYPE));
      b.Return();
    });
    std::mt19937 rng(100 + sh);
    for (int it = 0; it < 20; ++it) {
      vec128_t a = RandomVec(rng);
      test.Run([&](PPCContext* ctx) { ctx->v[4] = a; },
               [&](PPCContext* ctx) {
                 for (int j = 0; j < 8; ++j) {
                   INFO("sh " << sh << " lane " << j);
                   REQUIRE(SH(ctx->v[3], j) == int16_t(SH(a, j) >> sh));
                   REQUIRE(H(ctx->v[6], j) == uint16_t(H(a, j) << sh));
                 }
               });
    }
  }
}

namespace {
// vmrghb, vmrglb and every vsldoi shift, as the PPC frontend builds them.
std::vector<std::pair<const char*, vec128_t>> PermuteControls() {
  std::vector<std::pair<const char*, vec128_t>> out;
  out.push_back({"vmrghb", vec128b(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6,
                                   22, 7, 23)});
  out.push_back({"vmrglb", vec128b(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13,
                                   29, 14, 30, 15, 31)});
  static const char* names[16] = {
      "vsldoi0",  "vsldoi1",  "vsldoi2",  "vsldoi3",  "vsldoi4",  "vsldoi5",
      "vsldoi6",  "vsldoi7",  "vsldoi8",  "vsldoi9",  "vsldoi10", "vsldoi11",
      "vsldoi12", "vsldoi13", "vsldoi14", "vsldoi15"};
  for (int sh = 1; sh < 16; ++sh) {
    vec128_t c;
    for (int j = 0; j < 16; ++j) B(c, j) = uint8_t(sh + j);
    out.push_back({names[sh], c});
  }
  return out;
}
}  // namespace

TEST_CASE("VMX_FUZZ_PERMUTE_BYTES", "[instr][vmxfuzz]") {
  for (auto& [name, control] : PermuteControls()) {
    for (int zero_b = 0; zero_b < 2; ++zero_b) {
      vec128_t ctl = control;
      TestFunction test([ctl, zero_b](HIRBuilder& b) {
        Value* vb = zero_b ? b.LoadZeroVec128() : LoadVR(b, 5);
        StoreVR(b, 3,
                b.Permute(b.LoadConstantVec128(ctl), LoadVR(b, 4), vb,
                          INT8_TYPE));
        b.Return();
      });
      std::mt19937 rng(7);
      for (int it = 0; it < 20; ++it) {
        vec128_t a = RandomVec(rng);
        vec128_t c = zero_b ? vec128i(0) : RandomVec(rng);
        test.Run([&](PPCContext* ctx) { ctx->v[4] = a; ctx->v[5] = c; },
                 [&](PPCContext* ctx) {
                   vec128_t ctl_copy = ctl;
                   for (int j = 0; j < 16; ++j) {
                     int idx = B(ctl_copy, j) & 31;
                     uint8_t want = idx < 16 ? B(a, idx) : B(c, idx - 16);
                     INFO(name << " zero_b " << zero_b << " byte " << j);
                     REQUIRE(B(ctx->v[3], j) == want);
                   }
                 });
      }
    }
  }
}

TEST_CASE("VMX_FUZZ_VPKSHUS", "[instr][vmxfuzz]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 3,
            b.Pack(LoadVR(b, 4), LoadVR(b, 5),
                   PACK_TYPE_8_IN_16 | PACK_TYPE_IN_SIGNED |
                       PACK_TYPE_OUT_UNSIGNED | PACK_TYPE_OUT_SATURATE));
    b.Return();
  });
  std::mt19937 rng(3);
  for (int it = 0; it < kIters; ++it) {
    vec128_t a = (it & 1) ? RandomVec(rng) : RandomSmallH(rng);
    vec128_t c = (it & 2) ? RandomVec(rng) : RandomSmallH(rng);
    test.Run([&](PPCContext* ctx) { ctx->v[4] = a; ctx->v[5] = c; },
             [&](PPCContext* ctx) {
               for (int j = 0; j < 8; ++j) {
                 INFO("it " << it << " lane " << j);
                 REQUIRE(B(ctx->v[3], j) == SatU8(SH(a, j)));
                 REQUIRE(B(ctx->v[3], 8 + j) == SatU8(SH(c, j)));
               }
             });
  }
}

// Eight vectors live in host registers across a native helper call (a
// non-uniform VectorShl is emulated in C++ through the guest-to-host thunk).
TEST_CASE("VMX_FUZZ_LIVE_ACROSS_NATIVE_CALL", "[instr][vmxfuzz]") {
  TestFunction test([](HIRBuilder& b) {
    Value* live[8];
    for (int k = 0; k < 8; ++k) {
      live[k] = b.VectorAdd(LoadVR(b, 10 + k), LoadVR(b, 20 + k), INT16_TYPE,
                            0);
    }
    Value* shifted = b.VectorShl(LoadVR(b, 4), LoadVR(b, 5), INT16_TYPE);
    StoreVR(b, 3, shifted);
    for (int k = 0; k < 8; ++k) {
      StoreVR(b, 30 + k, b.VectorAdd(live[k], shifted, INT16_TYPE, 0));
    }
    b.Return();
  });
  std::mt19937 rng(4);
  for (int it = 0; it < 50; ++it) {
    vec128_t in[20];
    for (auto& v : in) v = RandomVec(rng);
    vec128_t a = RandomVec(rng);
    vec128_t s;
    for (int j = 0; j < 8; ++j) H(s, j) = uint16_t(j + 1);  // non-uniform
    test.Run(
        [&](PPCContext* ctx) {
          ctx->v[4] = a;
          ctx->v[5] = s;
          for (int k = 0; k < 8; ++k) {
            ctx->v[10 + k] = in[k];
            ctx->v[20 + k] = in[10 + k];
          }
        },
        [&](PPCContext* ctx) {
          for (int k = 0; k < 8; ++k) {
            for (int j = 0; j < 8; ++j) {
              uint16_t shifted = uint16_t(H(a, j) << (j + 1));
              uint16_t want =
                  uint16_t(H(in[k], j) + H(in[10 + k], j) + shifted);
              INFO("it " << it << " vec " << k << " lane " << j);
              REQUIRE(H(ctx->v[30 + k], j) == want);
            }
          }
        });
  }
}
