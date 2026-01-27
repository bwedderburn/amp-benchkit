## Sub-issue: BM83 EQ cycling triggers backend crash (ref @bwedderburn/BM83-ESP32-S3-Nextion#35)

### Summary
While rapidly cycling Bluetooth EQ modes on the Nextion UI, the BM83 stack sends a burst of `BT_EQ` tokens and AVRCP metadata requests. Shortly after a BLE disconnect, Thonny’s backend crashes with `ConnectionError: EOF` (see backend.log) and the process exits with code 1.

### Repro (from serial/back-end logs)
1. Cycle EQ through OFF → SOFT → BASS → … → USER repeatedly; serial shows `BM83 TX AA 00 03 1C <mode> 00`.
2. A BLE disconnect occurs (`[BLE] Disconnected`, `BTM_Status state=0x02/0x0F`), followed by reinit/pairing.
3. During reconnect, frequent `BT_EQ` tokens and AVRCP metadata (`op=0x1A/0x5D`) are interleaved.
4. Thonny backend throws `PROBLEM IN THONNY'S BACK-END: Exception while handling 'Run' (ConnectionError: EOF)` and stops.

### Hypothesis
- Backend likely closes unexpectedly when the UART stream floods with EQ/metadata responses during BLE reconnect.
- Unhandled EOF/disconnect in the run loop propagates to Thonny, aborting the session.

### Proposed fix
- Add defensive handling around the run/IO loop to catch `ConnectionError`/EOF during BM83 reconnect and gracefully reset or re-open the port instead of crashing.
- Optionally rate-limit EQ token handling during reconnect to avoid flooding.

### Notes
- See attached serial excerpt in issue body for full context.
- backend.log confirms the EOF originates from the Thonny backend during the crash.
