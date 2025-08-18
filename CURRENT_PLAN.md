# Memory Allocation Fix Implementation Plan
## macOS ARM64 Virtual Address Space Exhaustion Fix

### Problem Statement
The xenia-cpu-tests intermittently crash (3-10% of runs) with SIGSEGV during the ADD_I16 test due to virtual address space exhaustion. Investigation revealed:

- **Root Cause**: macOS ARM64 implementation allocates one contiguous 4.5GB block per test
- **Other Platforms**: Windows/Linux allocate individual views (~2.5GB total with overlapping regions)
- **Impact**: After 10-30 test runs, virtual address space is exhausted causing allocation failures
- **Symptom**: `page_table_.resize()` receives nullptr from operator new, leading to null pointer dereference

### Current Status
- Branch: `arm64-macos-consolidated`
- Recent fixes completed: PACK/UNPACK_FLOAT16_2 Xbox 360 sentinel value handling
- Test status: All tests pass when they don't crash from memory exhaustion

---

## Implementation Plan

### Phase 1: Preparation
- [x] Analyze current implementation differences
- [x] Understand memory mapping requirements
- [x] Create implementation plan
- [ ] Backup current state before changes

### Phase 2: Code Implementation

#### Changes Required
**File**: `src/xenia/memory.cc`
- Remove lines 284-327 (macOS-specific contiguous allocation)
- Use standard implementation for all platforms
- Ensure individual view mapping works correctly on macOS ARM64

#### Expected Code Change
```cpp
int Memory::MapViews() {
    // Remove #if XE_PLATFORM_MAC && defined(__aarch64__)
    // Use same implementation for ALL platforms
    for (size_t n = 0; n < xe::countof(map_info); n++) {
        // Map each view individually
        views_.all_views[n] = MapFileView(...);
    }
}
```

### Phase 3: Build & Compilation
- [ ] Clean build: `./xb clean`
- [ ] Build with Checked config: `./xb build --config Checked`
- [ ] Verify all test targets built successfully

### Phase 4: Testing Protocol

#### 4.1 Basic Functionality Tests
- [ ] Run xenia-cpu-ppc-tests once
- [ ] Run xenia-cpu-tests once
- [ ] Verify PACK_FLOAT16_2 tests pass
- [ ] Verify UNPACK_FLOAT16_2 tests pass

#### 4.2 Memory Usage Verification
- [ ] Measure peak memory before fix (4.5GB expected)
- [ ] Measure peak memory after fix (2.5GB expected)
- [ ] Confirm ~40% reduction in memory usage

#### 4.3 Stress Testing
- [ ] Run 100 consecutive xenia-cpu-tests iterations
- [ ] Monitor for SIGSEGV crashes
- [ ] Expected: 0% crash rate (was 3-10%)

#### 4.4 Specific Test Verification
- [ ] ADD_I8 test passes
- [ ] ADD_I16 test passes (this was crashing)
- [ ] ADD_I32 test passes
- [ ] ADD_I64 test passes
- [ ] All PACK_* tests pass
- [ ] All UNPACK_* tests pass

### Phase 5: Validation Criteria

#### Must Pass
1. Zero crashes in 100+ consecutive runs
2. All existing tests continue to pass
3. Memory usage reduced by ~40%
4. No performance regression

#### Monitoring Points
- Virtual address allocation patterns
- Memory reclamation after test completion
- Peak memory usage per test
- Test execution time

### Phase 6: Risk Mitigation

#### Identified Risks
1. **Non-contiguous memory breaks assumptions**
   - Mitigation: Thorough testing of all memory operations
   - Rollback: Git revert if critical issues found

2. **Performance impact from individual mappings**
   - Mitigation: Benchmark before/after
   - Acceptable: <10% slowdown for stability

3. **Platform-specific issues**
   - Mitigation: Test on multiple macOS versions if possible
   - Focus: Apple Silicon (M1/M2/M3)

### Phase 7: Documentation & Commit

#### Commit Message Format
```
[Memory] Fix virtual address space exhaustion on macOS ARM64

Problem: Tests crash intermittently with SIGSEGV due to virtual 
address exhaustion. macOS allocates 4.5GB contiguous blocks while 
other platforms use 2.5GB with individual mappings.

Solution: Use individual view mappings on all platforms.

Results:
- Memory: 4.5GB → 2.5GB per test
- Crashes: 3-10% → 0%
- All tests pass

Testing: 100+ consecutive runs verified
```

---

## Progress Tracking

### Completed
- [x] Root cause analysis
- [x] Platform comparison
- [x] Implementation planning

### In Progress
- [ ] Code implementation
- [ ] Testing
- [ ] Verification

### Todo
- [ ] Final validation
- [ ] Documentation
- [ ] Commit changes

---

## Test Commands Reference

```bash
# Build
./xb clean
./xb build --config Checked

# Single test run
./build/bin/Mac/Checked/xenia-cpu-tests
./build/bin/Mac/Checked/xenia-cpu-ppc-tests

# Stress test (100 runs)
for i in {1..100}; do
    echo "Run $i"
    if ! ./build/bin/Mac/Checked/xenia-cpu-tests > /tmp/test_$i.log 2>&1; then
        echo "FAILED on run $i"
        tail -50 /tmp/test_$i.log
        break
    fi
    echo "OK"
done

# Memory monitoring
vm_stat | grep "Pages free"
```

---

## Rollback Plan
If issues arise:
1. `git diff src/xenia/memory.cc` - Review changes
2. `git checkout -- src/xenia/memory.cc` - Revert
3. Consider alternative approaches:
   - Reduce test memory size
   - Share Memory instance across tests
   - Implement memory pool management

---

## Success Metrics
- **Primary**: 0% crash rate in 100+ runs
- **Secondary**: 40% memory reduction (4.5GB → 2.5GB)
- **Tertiary**: No performance regression (±10%)

---

## Notes
- Platform difference discovered: macOS uses contiguous allocation unnecessarily
- Windows/Linux approach proven stable over years
- Fix aligns all platforms to same implementation
- Side benefit: Easier maintenance with unified code path