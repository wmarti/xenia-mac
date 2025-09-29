Boot Test Harness

Quick start:
1) Generate manifest:
   python3 tools/boot_test/generate_manifest.py --games-root /Users/admin/Documents/X360-Games

2) Run tests:
   python3 tools/boot_test/run_boot_tests.py --xenia ./build/bin/Release/xenia

3) Compare runs:
   python3 tools/boot_test/compare_runs.py tools/boot_test/runs/<baseline> tools/boot_test/runs/<current>

Notes:
- You can override the xenia binary path using --xenia or XENIA_BIN.
- Logs are stored per run in tools/boot_test/runs/<timestamp_commit>/logs.
- Edit tools/boot_test/milestones.json to tune progress detection.
