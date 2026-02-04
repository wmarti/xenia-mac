project_root = "../../../.."
include(project_root.."/tools/build")

test_suite("xenia-cpu-tests", project_root, ".", {
  links = {
    "capstone",
    "fmt",
    "imgui",
    "xenia-apu",  -- Explicitly link xenia-apu (transitively required by xenia-kernel)
    "xenia-base",
    "xenia-core",
    "xenia-cpu",
    "xenia-gpu",  -- Required by xenia-kernel (video syscalls)
    "xenia-hid-skylander",

    -- TODO(benvanik): cut these dependencies?
    "xenia-kernel",
    "xenia-ui", -- needed by xenia-base
    "xenia-patcher",
  },
  filtered_links = {
    {
      filter = 'architecture:x86_64',
      links = {
        "xenia-cpu-backend-x64",
      },
    },
    {
      filter = 'architecture:ARM64',
      links = {
        "xenia-cpu-backend-a64",
      },
    },
  },
})

-- xenia-kernel links to xenia-apu, which needs SDL on Linux
apu_transitive_deps()

-- macOS requires additional frameworks for Metal/UI
filter("system:macosx")
  links({
    "QuartzCore.framework",
    "Metal.framework",
    "Foundation.framework",
    "AppKit.framework",
  })
filter({"system:macosx", "architecture:x86_64"})
  linkoptions({
    "-Wl,-pagezero_size,0x1000",
  })
filter({})
