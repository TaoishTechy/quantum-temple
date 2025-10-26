# Releasing Quantum Temple

1. Update CHANGELOG.md
2. Tag SemVer
   git tag -a v0.2.0 -m "v0.2.0: CPTP fix, expm_multiply, tests"
   git push origin v0.2.0
3. CI will:
   - run tests
   - run a short benchmark
   - attach `benchmark_bundle.tgz` to the GitHub Release
4. Publish Release notes (auto-generated), add highlights.
