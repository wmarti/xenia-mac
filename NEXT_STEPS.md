# NEXT_STEPS (updated 2026-01-02)

This doc tracks the null GPU app bring-up for A64 debugging on macOS.

## Phase 0: Null GPU App Bring-Up (ACTIVE)

### Problem Description
We need a dedicated worktree and configuration to run `xenia-app` with the
null graphics backend on macOS so we can load XEX files and focus on A64 CPU
backend correctness without GPU noise.

### Root Cause Analysis
- The default configuration uses `gpu=any`, which selects Metal on macOS.
- Existing docs and targets focus on Metal trace replay rather than app usage.
- There is no worktree-local runtime config or build log workflow for
  `xenia-app` under the null backend.

### Implementation Checklist
- [x] Create worktree in `scratch/worktrees/null-app-mac`.
- [x] Update `AGENTS.md` for null GPU + Metal UI + `xenia-app` focus.
- [x] Add worktree-local config in `scratch/` for `gpu=null` and storage.
- [x] Link `third_party/` to the main worktree to reuse existing submodules.
- [x] Build `xenia-app` with logs under `scratch/logs/`.
      Log: `scratch/logs/build.log`.
- [ ] Document the launch command and capture an initial run log in `scratch/`.

### Reference Information
- Null backend setup: `src/xenia/gpu/null/null_graphics_system.cc`
- GPU selection CVar: `src/xenia/app/xenia_main.cc`
- Build: `./xb build --target=xenia-app`
- Run: `./build/bin/Mac/Checked/xenia-app --config=scratch/config/null-app-mac.config.toml <path/to/game.xex>`

## Phase 1: A64 + Memory Canary Patch Intake (PLANNED)

### Problem Description
We need to pull select ARM64/A64 backend and memory fixes from the canary
worktrees to improve A64 stability on macOS without dragging unrelated
Android/Vulkan changes.

### Root Cause Analysis
- A64 indirection table still has edge cases on ARM64 (high addresses,
  fixed mapping failures, and table unmap sizing).
- POSIX memory mapping and page-size assumptions can break on 16 KiB pages.
- A64 emitter/backend has correctness gaps for constant vector conversions and
  resolve-function fallback paths.

### Implementation Checklist
- [x] Apply canary patches (manual apply due to worktree drift; vastcpy changes
      in 0004 not present in this branch):
  1) `git -C /Users/wmarti/Documents/xenia-mac/scratch/worktrees/null-app-mac apply /Users/wmarti/Documents/xenia-mac/scratch/patches/canary-a64/0001-A64-indirection-table-ARM64.patch`
  2) `git -C /Users/wmarti/Documents/xenia-mac/scratch/worktrees/null-app-mac apply /Users/wmarti/Documents/xenia-mac/scratch/patches/canary-a64/0002-Base-validate-posix-mappings.patch`
  3) `git -C /Users/wmarti/Documents/xenia-mac/scratch/worktrees/null-app-mac apply /Users/wmarti/Documents/xenia-mac/scratch/patches/canary-a64/0003-A64-uncommitted-changes.patch`
  4) `git -C /Users/wmarti/Documents/xenia-mac/scratch/worktrees/null-app-mac apply /Users/wmarti/Documents/xenia-mac/scratch/patches/canary-a64/0004-memory-uncommitted-changes.patch`
- [ ] Cross-check A64/macOS changes against x64/Windows reference behavior
      (use x64 backend and D3D12/Vulkan paths for correctness).
- [x] Run base tests (checked config) before game validation.
      Log: `scratch/logs/test-base-run.log`
- [x] Run CPU tests (checked config) before game validation.
      Log: `scratch/logs/test-cpu-run.log`
- [x] Run PPC CPU tests (checked config) before game validation.
      Log: `scratch/logs/test-ppc-run.log`
- [x] Populate PPC test binaries from canary worktree and build native thunks:
      `cp -a /Users/wmarti/Documents/xenia-mac/scratch/worktrees/`
      `xenia-canary-rebase/src/xenia/cpu/ppc/testing/bin/.`
      `src/xenia/cpu/ppc/testing/bin/`
      `powerpc-none-elf-*` tools from canary binutils were used to build
      `ppc_testing_native_thunks.{o,bin,dis,map}`.
- [x] Review test results:
      `grep -E "(PASSED|FAILED|passed|failed)" scratch/logs/test-base-run.log`
      `tail -20 scratch/logs/test-base-run.log`
      `grep -E "(PASSED|FAILED|passed|failed)" scratch/logs/test-cpu-run.log`
      `tail -20 scratch/logs/test-cpu-run.log`
      `grep -E "(PASSED|FAILED|passed|failed)" scratch/logs/test-ppc-run.log`
      `tail -20 scratch/logs/test-ppc-run.log`
      Results: base/cpu/ppc tests passed.
      Note: `xb test` on macOS only accepts one scheme at a time; running
      multiple targets fails with `-scheme` duplication. Use manual build/run
      per target if needed.
- [ ] Decide whether to apply optional PPC extern BLR stub patch:
  `git -C /Users/wmarti/Documents/xenia-mac/scratch/worktrees/null-app-mac apply /Users/wmarti/Documents/xenia-mac/scratch/patches/canary-a64/0005-optional-ppc-extern-blr-stubs.patch`
- [ ] Rebuild `xenia-app` (after tests pass) and rerun the same XEX to validate.

### Reference Information
- Patch set location: `scratch/patches/canary-a64/`
- A64 code cache/emitter: `src/xenia/cpu/backend/a64/`
- POSIX memory mapping: `src/xenia/base/memory_posix.cc`

## Phase 1B: A64 Perf Regression After Patch Intake (ACTIVE)

### Problem Description
PPC CPU tests became dramatically slower after patch intake versus the
baseline `xenia-mac` worktree.

Now seeing an `EXC_BAD_ACCESS` in `XThread::Execute` on macOS when logging
`thread_->system_id()`, indicating `thread_` is null during thread start.

### Root Cause Analysis
`AllocFixed` no longer falls back to OS-chosen addresses on macOS ARM64, so
`A64Emitter::PlaceConstData()` looped through many low fixed addresses blocked
by PAGEZERO. That caused repeated failed `mmap` attempts per backend init and
slowed test startup.

`thread_` is likely null due to a race in the current mac threading
implementation: `Thread::Create` sets `create_suspended=true`, but the thread
can run `start_routine` before the creator assigns `thread_` and before
`thread_suspend` is applied. Canary’s `threading_mac.cc` gates execution so
start routines wait until resume, preventing `thread_`-null usage in
`XThread::Execute`.

### Implementation Checklist
- [x] Confirm patch set already applied (`git apply --check` fails).
- [x] Compare `instr_vspltw` timing against baseline.
- [x] Use OS-chosen const-data mapping on macOS ARM64 to avoid fixed-address
      retry loops.
- [x] Cache `IsWritableExecutableMemorySupported()` result.
- [x] Rebuild `xenia-cpu-ppc-tests` and re-measure `instr_vspltw` (fast path
      restored after warm run).
- [x] Run base/cpu/ppc tests (logs under `scratch/logs/`).
- [ ] Compare full PPC test runtime vs baseline (optional sanity).
- [x] Port canary `src/xenia/base/threading_mac.cc` to fix thread-start race.
- [ ] Rebuild base/cpu/ppc tests and rerun `xenia-app` to confirm crash fixed.
- [ ] If needed, enable instruction tracing (see `docs/instruction_tracing.md`).

Note: base/cpu/ppc tests were rerun after the threading port; all passed.

### Reference Information
- Timing logs: `scratch/logs/time-ppc-tests-*.log`
- Const data placement: `src/xenia/cpu/backend/a64/a64_emitter.cc`
- MAP_JIT probe: `src/xenia/base/memory_posix.cc`
- Threading macOS (canary):
  `scratch/worktrees/xenia-canary-rebase/src/xenia/base/threading_mac.cc`
- Instruction tracing: `docs/instruction_tracing.md`

## Phase 2: Threading macOS Canary Intake (DONE)

### Problem Description
The canary worktree contains a redesigned `threading_mac.cc` based on the POSIX
implementation. We needed to port it to fix macOS thread behavior issues
without regressing JIT or UI stability.

### Root Cause Analysis
- Current mac threading implementation differs from POSIX and uses Mach thread
  suspend/resume logic that may interact poorly with A64 codegen and signal
  handling.
- Canary’s implementation aligns with POSIX threading behavior but is a large
  change and has to be vetted for correctness on macOS (signal usage, suspend,
  callback handling).

### Implementation Checklist
- [x] Sync `threading_mac.cc` from canary (direct port; no patch file).
- [x] Resolve the `EventInfo` override build break.
- [ ] Rebuild `xenia-app` and rerun the same XEX to validate.

### Reference Information
- Canary source: `scratch/worktrees/xenia-canary-rebase/src/xenia/base/threading_mac.cc`
- Current source: `src/xenia/base/threading_mac.cc`

## Phase 3: Thread Exit Lifetime Crash (ACTIVE)

### Problem Description
On macOS, `XThread::Exit` sometimes triggers `std::system_error: mutex lock
failed: Invalid argument` inside `xe::threading::PosixCondition<Thread>::Terminate`.
Logs show `Removed handle` for the exiting thread immediately before
`XThread::Exit` calls `Thread::Exit`.

### Root Cause Analysis
`XThread::Exit` calls `ReleaseHandle` before `Thread::Exit`. If the thread has
no remaining `object_ref` owners, `ReleaseHandle` removes the handle and can
delete the `XThread` object, destroying `thread_`. `Thread::Exit` then calls
`current_thread_->Terminate`, which locks a mutex inside the now-destroyed
`PosixThread` object, producing `EINVAL`.

### Implementation Checklist
- [ ] Confirm handle refcount + object lifetime at `ReleaseHandle`
      (instrument `ObjectTable::ReleaseHandle` if needed).
- [ ] Compare `XThread::Exit` lifetime management vs canary/x64 behavior.
- [ ] Decide on a safe lifetime strategy (delay handle release, add a self
      retain until exit, or relocate cleanup to thread-exit callback).
- [ ] Rebuild base/cpu/ppc tests and reproduce to validate fix.

### Reference Information
- `XThread::Exit`: `src/xenia/kernel/xthread.cc`
- Handle release: `src/xenia/kernel/util/object_table.cc`
- Thread exit path: `src/xenia/base/threading_mac.cc`

## Phase 4: macOS MMIO SIGBUS Handling (ACTIVE)

### Problem Description
Guest MMIO accesses in the 0x7Fxxxxxx alias range crash with `EXC_BAD_ACCESS`
on macOS instead of trapping into the MMIO handler. The crash shows a host
address in the 0x37f... range immediately after `RtlInitializeCriticalSection`.

### Root Cause Analysis
The macOS exception handler only installs `SIGSEGV` handling. On macOS, access
violations from `PROT_NONE`-backed MMIO regions can raise `SIGBUS`, so the
MMIO exception handler never runs and the process aborts.

### Implementation Checklist
- [x] Install and restore `SIGBUS` alongside `SIGSEGV` in the mac exception
      handler.
- [ ] Rebuild `xenia-app` and confirm MMIO faults are handled (no 0x7F alias
      crash).
- [ ] Capture updated logs in `scratch/logs/` and validate stability.
- [x] Rerun base/cpu/ppc tests to ensure no regressions.

### Reference Information
- mac exception handler: `src/xenia/base/exception_handler_mac.cc`
- Memory mapping log: `scratch/logs/app.log`

## Context Summary (carry into next session)

- Patch set prepared at `scratch/patches/canary-a64/`:
  - `0001-A64-indirection-table-ARM64.patch` (commit 11bd8616b)
  - `0002-Base-validate-posix-mappings.patch` (commit 0563e23f2)
  - `0003-A64-uncommitted-changes.patch` (ResolveFunction 64-bit handling,
    A64 AV log, BTI prolog, constant vector convert fix, indirection logging)
  - `0004-memory-uncommitted-changes.patch` (page-size aware vastcpy,
    stricter MAP_FIXED_NOREPLACE behavior)
  - `0005-optional-ppc-extern-blr-stubs.patch` (optional)
- `0001`-`0004` applied manually in this worktree; `0004` vastcpy changes not
  applicable because `src/xenia/base/memory.cc` doesn't include that logic.
- A mistaken apply to the main repo was reverted; no changes remain there.
- `third_party/` is symlinked in this worktree to avoid re-fetching submodules.
- Build completed in this worktree; log at `scratch/logs/build.log`.
- A64 perf regression traced to const-data fixed mapping retries on macOS.
  `PlaceConstData` now uses OS-chosen mappings on ARM64 macOS and
  `IsWritableExecutableMemorySupported()` is cached.
- Base/CPU/PPC tests passed:
  `scratch/logs/test-base-run.log`, `scratch/logs/test-cpu-run.log`,
  `scratch/logs/test-ppc-run.log`.
