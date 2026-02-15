# Next Tasks Checklist

- [x] Verify llama-server binary path and model path; run `scripts/launch_server.py` to confirm clean boot.
- [x] Validate JSON mode using `medgemma_client.py` with `generate_strict_json()`; confirm parsed output.
- [x] Add comprehensive README with setup, usage, API reference, and team integration guide.
- [x] Create verification test suite with 10 test scenarios and output logging.
- [x] Create smoke test script for quick pre-deployment validation.
- [ ] Decide production bind strategy (`0.0.0.0` vs `127.0.0.1`) and add basic auth or reverse proxy if LAN exposure is required.
- [ ] Optional: Update test suite timeouts for long-running tests and re-run verification.
