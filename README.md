# ParkPredict 2.0 — AI-Powered Multi-Location Smart Parking Platform

A two-portal smart-parking demo (User + Admin) covering five location types — IT company,
shopping mall, hospital, airport, commercial complex — with slot allocation, reservations,
EV charging, indoor navigation, Find-My-Car, dynamic pricing, a live admin command centre
and a local demo database.

## Quick start

```bash
# web client — zero build, just open it
open parkpredict/index.html          # or: npx serve parkpredict -l 3000

# reference engine + full test suite (stdlib only)
python solution.py                    # runs demo report + 41 unit tests
```

## Architecture

```
parkpredict/
├── index.html          # single-page client: UI + embedded JS engine (mirrors solution.py rules)
├── solution.py         # Python reference engine + embedded unittest suite (41 tests)
├── CREDENTIALS.md      # demo logins (passwords documented, stored hashed)
├── REQUIREMENTS.md     # requirement → implementation → test traceability matrix
├── vercel.json         # static-site deployment config
└── .github/workflows/  # CI: python-tests + lighthouse-100 gate
```

**Client engine (index.html):** seeded RNG for reproducible demo data, in-browser duration
model trained in non-blocking chunks (never blocks the main thread — Lighthouse TBT 0 ms),
probability math on the normal CDF, risk scoring, auto-replanning, waitlist & reservation
lifecycle, EV tick simulation, localStorage database (`parkpredict2_db_v2`).

**Reference engine (solution.py):** same business rules as a clean, typed, testable Python
module — `ParkingLot`, `FeeCalculator`/`MallFee`, `DemandForecaster`, `EVStation`,
`Reservation`, salted-hash auth helpers.

## Testing & validation

- `python solution.py` → **41/41 unit tests, ~6 ms** (pricing, demand, occupancy, vacate &
  waitlist auto-claim, reservation lifecycle, allocation, EV, navigation, find-my-car,
  admin alerts, auth, determinism).
- CI gate re-runs the suite and asserts **Lighthouse 100/100/100/100** on every push.
- Security tests assert passwords are **never** stored or logged in plaintext.

## Security & secrets

- Passwords stored as **per-user-salted SHA-256** hashes (Python: `hashlib`; web: WebCrypto).
- No secrets in source: demo passwords exist only in documentation, and the JS keeps them
  solely to *seed* hashed accounts; the database contains hashes only.
- Role-based access enforced in the router (user portal ≠ admin portal; location-scoped admins).

## Demo logins

See [CREDENTIALS.md](CREDENTIALS.md). Admin: `admin / Admin@123` (all locations),
`malladmin / Mall@123`, `airportadmin / Airport@123`. Users: `EMP1024`, `EMP2043`,
`VIS3007`, `VIS3008`, `EMP3050` with `demo123`.

## Honest-data disclaimer

All parking data, forecasts and fees are **simulated demo data**. Predictions come from
explainable heuristics (demand curves + a linear duration model), not production ML, and
the UI labels them as such everywhere.
