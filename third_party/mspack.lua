group("third_party")
project("mspack")
  uuid("0881692A-75A1-4E7B-87D8-BB9108CEDEA4")
  kind("StaticLib")
  language("C++")
  links({
    "xenia-base",
  })
  defines({
    "HAVE_CONFIG_H",
  })
  includedirs({
      "mspack",
      "%{wks.location}/../third_party/mspack",
  })
  externalincludedirs({
      "mspack",
      -- libmspack uses angle-bracket local includes (<config.h>, <system.h>,
      -- etc.), so expose the same directory as external/system includes.
      "%{wks.location}/../third_party/mspack",
  })
  files({
      "mspack/logging.cc",
      "mspack/lzxd.c",
      "mspack/system.c",
  })
