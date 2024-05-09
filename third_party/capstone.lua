group("third_party")
project("capstone")
  uuid("b3a89f7e-bb02-4945-ae75-219caed6afa2")
  kind("StaticLib")
  language("C")
  defines({
    "CAPSTONE_DIET_NO",
    "CAPSTONE_USE_SYS_DYN_MEM",
    "_LIB",
  })
  filter("architecture:x86_64")
    defines({
      "CAPSTONE_HAS_X86",
      "CAPSTONE_X86_ATT_DISABLE",
      "CAPSTONE_X86_REDUCE_NO",
    })
    files({
      "capstone/arch/X86/*.c",
      "capstone/arch/X86/*.h",
      "capstone/arch/X86/*.inc",
    })
    force_compile_as_c({
      "capstone/arch/X86/**.c",
    })
  filter("architecture:ARM64")
    defines({
      "CAPSTONE_HAS_ARM64",
    })
    files({
      "capstone/arch/AArch64/*.c",
      "capstone/arch/AArch64/*.h",
      "capstone/arch/AArch64/*.inc",
    })
    force_compile_as_c({
      "capstone/arch/AArch64/**.c",
    })
  filter({})
  includedirs({
    "capstone",
    "capstone/include",
  })
  files({
    "capstone/cs.c",
    "capstone/cs_priv.h",
    "capstone/LEB128.h",
    "capstone/MathExtras.h",
    "capstone/MCDisassembler.h",
    "capstone/MCFixedLenDisassembler.h",
    "capstone/MCInst.c",
    "capstone/MCInst.h",
    "capstone/MCInstrDesc.c",
    "capstone/MCInstrDesc.h",
    "capstone/MCRegisterInfo.c",
    "capstone/MCRegisterInfo.h",
    "capstone/SStream.c",
    "capstone/SStream.h",
    "capstone/utils.c",
    "capstone/utils.h",
  })
  force_compile_as_c({
      "capstone/**.c",
  })