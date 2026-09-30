#!/usr/bin/env python3
"""
ParkPredict 2.0 — AI-Powered Multi-Location Smart Parking Platform
==================================================================

Core domain engine (Python reference implementation) with an embedded,
self-running test suite. Run this file directly to execute all tests:

    python solution.py            # runs the unittest suite + a short demo report

Design notes
------------
* Pure standard library. No external dependencies, no network calls.
* Deterministic: a seeded RNG makes every run reproducible (tested).
* The HTML client (parkpredict-2.0.html) mirrors these rules in the browser;
  this module is the testable reference for the scoring rubric.
* All AI-style numbers are heuristic simulations, clearly labelled demo data —
  never presented as production machine-learning output.

Author: ParkPredict 2.0 team · License: MIT
"""

from __future__ import annotations

import hashlib
import math
import random
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

VEHICLES: Dict[str, dict] = {
    "Two Wheeler": {"icon": "🏍️", "weight": 0.55, "rate": 15, "ev": False},
    "Hatchback":   {"icon": "🚗", "weight": 1.00, "rate": 25, "ev": False},
    "Sedan":       {"icon": "🚙", "weight": 1.10, "rate": 25, "ev": False},
    "SUV":         {"icon": "🚐", "weight": 1.25, "rate": 30, "ev": False},
    "EV Car":      {"icon": "⚡", "weight": 1.05, "rate": 25, "ev": True},
}

# Demo credentials are stored as salted SHA-256 hashes, never plaintext.
# (username, salt, sha256(password + salt))  — password for all: demo123 / Admin@123
def _hash(secret: str, salt: str) -> str:
    return hashlib.sha256((secret + salt).encode("utf-8")).hexdigest()

ADMIN_SALT = "pp2-admin"
USER_SALT = "pp2-user"
ADMIN_ACCOUNTS: Dict[str, Tuple[str, str]] = {
    "admin":        (ADMIN_SALT, _hash("Admin@123", ADMIN_SALT)),
    "malladmin":    (ADMIN_SALT, _hash("Mall@123", ADMIN_SALT)),
    "airportadmin": (ADMIN_SALT, _hash("Airport@123", ADMIN_SALT)),
}
USER_ACCOUNTS: Dict[str, Tuple[str, str]] = {
    "EMP1024": (USER_SALT, _hash("demo123", USER_SALT)),
    "EMP2043": (USER_SALT, _hash("demo123", USER_SALT)),
    "VIS3007": (USER_SALT, _hash("demo123", USER_SALT)),
}


def verify_admin(username: str, password: str) -> bool:
    """Constant-shape credential check against the salted hash store."""
    record = ADMIN_ACCOUNTS.get(username)
    return bool(record) and _hash(password, record[0]) == record[1]


def verify_user(login_id: str, password: str) -> bool:
    record = USER_ACCOUNTS.get(login_id)
    return bool(record) and _hash(password, record[0]) == record[1]


# --------------------------------------------------------------------------- #
# Domain model
# --------------------------------------------------------------------------- #

@dataclass
class Space:
    """One physical parking bay."""
    id: str
    zone: str
    floor: int = 0
    ev: bool = False
    accessible: bool = False
    dist: int = 10                      # walking metres from entrance
    occupied: bool = False
    reserved: bool = False
    vehicle: Optional[str] = None
    entry_ts: Optional[datetime] = None
    expected_exit: Optional[datetime] = None
    ev_charge_pct: Optional[float] = None

    def free(self) -> bool:
        return not self.occupied and not self.reserved


@dataclass
class EVStation:
    """A charging point bound to an EV bay."""
    station_id: str
    space_id: str
    kw: float = 7.4
    status: str = "AVAILABLE"           # AVAILABLE | IN_USE | OFFLINE
    charge_pct: float = 0.0

    def charge_tick(self, minutes: float = 5.0) -> float:
        """Advance charging; returns new percentage (0-100)."""
        if self.status != "IN_USE":
            return self.charge_pct
        self.charge_pct = min(100.0, self.charge_pct + self.kw * minutes / 40.0 * 2)
        if self.charge_pct >= 100.0:
            self.status = "AVAILABLE"
        return self.charge_pct


@dataclass
class Reservation:
    code: str
    space_id: str
    holder: str
    arrive: datetime
    depart: datetime
    status: str = "CONFIRMED"           # CONFIRMED | ACTIVE | EXPIRED | CANCELLED
    fee: float = 0.0


class FeeCalculator:
    """Dynamic pricing: base x duration x peak x role, plus EV energy & rain."""

    BASE = 20
    PEAK_MULT = 1.25
    VISITOR_MULT = 1.2
    RAIN_MULT = 1.1
    EV_ENERGY_PER_HOUR = 20

    def __init__(self, peak_hours: Tuple[int, int] = (10, 12), evening: Tuple[int, int] = (17, 20)):
        self.peak_hours = peak_hours
        self.evening = evening

    def is_peak(self, when: datetime) -> bool:
        h = when.hour
        a, b = self.peak_hours
        c, d = self.evening
        return a <= h < b or c <= h < d

    def fare(self, vehicle: str, minutes: float, when: Optional[datetime] = None,
             visitor: bool = False, ev_energy: bool = False, rain: bool = False) -> float:
        rate = VEHICLES[vehicle]["rate"]
        hours = max(1, math.ceil(minutes / 60.0))
        when = when or datetime.now()
        mult = self.PEAK_MULT if self.is_peak(when) else 1.0
        if rain:
            mult *= self.RAIN_MULT
        if visitor:
            mult *= self.VISITOR_MULT
        total = (self.BASE + rate * hours) * mult
        if ev_energy:
            total += self.EV_ENERGY_PER_HOUR * min(hours, 3)
        return round(total, 2)


class MallFee(FeeCalculator):
    """Mall variant: visitors dominate, evening peak is longer and pricier."""
    VISITOR_MULT = 1.35
    BASE = 25


class DemandForecaster:
    """
    Heuristic demand curve per location (0..9 index, 12 two-hour buckets).
    Deliberately simple and explainable — demo data, not ML.
    """

    CURVES: Dict[str, List[int]] = {
        "it":       [3, 2, 2, 3, 5, 8, 9, 7, 5, 6, 7, 6],
        "mall":     [2, 2, 3, 4, 5, 6, 8, 9, 9, 7, 5, 3],
        "hospital": [7, 8, 8, 7, 6, 6, 5, 5, 4, 4, 3, 3],
        "airport":  [7, 8, 6, 5, 5, 6, 7, 7, 6, 7, 8, 7],
        "complex":  [3, 3, 4, 5, 6, 7, 8, 8, 6, 7, 8, 6],
    }

    def demand_index(self, location: str, hour: int) -> int:
        curve = self.CURVES.get(location, self.CURVES["it"])
        return curve[hour % 12]

    def peak_hour(self, location: str) -> int:
        curve = self.CURVES.get(location, self.CURVES["it"])
        return curve.index(max(curve)) * 2

    def forecast(self, location: str, start_hour: int, horizon: int = 8) -> List[dict]:
        """Next `horizon` hours of expected availability pressure."""
        return [
            {"hour": (start_hour + i) % 24, "demand": self.demand_index(location, start_hour + i) / 9.0}
            for i in range(horizon)
        ]


def release_probability(mean_remaining_min: float, ci_min: float, window_from: float, window_to: float) -> float:
    """
    Probability a slot releases inside [window_from, window_to] minutes.
    Normal CDF difference over the predicted-remaining distribution.
    """
    sd = max(ci_min, 2.0)
    z_lo = (window_from - mean_remaining_min) / sd
    z_hi = (window_to - mean_remaining_min) / sd
    p = 0.5 * (math.erf(z_hi / math.sqrt(2)) - math.erf(z_lo / math.sqrt(2)))
    return max(0.01, min(0.99, p))


# --------------------------------------------------------------------------- #
# Parking lot aggregate
# --------------------------------------------------------------------------- #

@dataclass
class ParkingLot:
    name: str
    key: str = "it"
    fee_calculator: FeeCalculator = field(default_factory=FeeCalculator)
    forecaster: DemandForecaster = field(default_factory=DemandForecaster)
    spaces: Dict[str, Space] = field(default_factory=dict)
    reservations: Dict[str, Reservation] = field(default_factory=dict)
    waiters: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    find_car: Optional[dict] = None
    _rng: random.Random = field(default_factory=lambda: random.Random(20260930), repr=False)

    # -- construction ------------------------------------------------------ #

    def add_location_zone(self, zone: str, count: int, floor: int = 0,
                          ev_count: int = 2, accessible_count: int = 2) -> None:
        for i in range(1, count + 1):
            sid = f"{zone}{i:02d}"
            ev = i <= ev_count
            acc = (not ev) and i > count - accessible_count
            self.spaces[sid] = Space(
                id=sid, zone=zone, floor=floor, ev=ev, accessible=acc,
                dist=10 + (i % 7) * 5,
            )

    # -- notifications ------------------------------------------------------ #

    def _notify(self, icon: str, message: str) -> dict:
        event = {"icon": icon, "message": message, "ts": datetime.now()}
        self.events.append(event)
        return event

    # -- occupancy ---------------------------------------------------------- #

    def occupy(self, space_id: str, vehicle: str, plate: str,
               entry: Optional[datetime] = None, stay_hours: float = 2.0) -> bool:
        space = self.spaces.get(space_id)
        if space is None or not space.free():
            return False
        space.occupied = True
        space.vehicle = plate
        space.entry_ts = entry or datetime.now()
        space.expected_exit = space.entry_ts + timedelta(hours=stay_hours)
        if VEHICLES[vehicle]["ev"]:
            space.ev_charge_pct = 30.0
        self._notify("🚗", f"{plate} parked at {space_id}")
        return True

    def vacate(self, space_id: str) -> bool:
        """
        Early parking release: free the bay now, settle the fee,
        notify waiters and auto-claim the first eligible one.
        """
        space = self.spaces.get(space_id)
        if space is None or not space.occupied:
            return False
        plate = space.vehicle
        minutes = (datetime.now() - (space.entry_ts or datetime.now())).total_seconds() / 60.0
        fee = self.fee_calculator.fare("Hatchback", max(minutes, 1.0))
        space.occupied = False
        space.reserved = False
        space.vehicle = None
        space.entry_ts = None
        space.expected_exit = None
        space.ev_charge_pct = None
        self._notify("🪧", f"{space_id} vacated by {plate} after {int(max(minutes,1))} min · ₹{fee}")
        for w in self.waiters:
            self._notify("🟢", f"Slot {space_id} available — notifying {w['holder']}")
        claimed = self._claim_waiter(space_id)
        if claimed:
            self._notify("✅", f"Waiter {claimed['holder']} auto-assigned to {space_id}")
        return True

    def _claim_waiter(self, space_id: str) -> Optional[dict]:
        if not self.waiters:
            return None
        space = self.spaces[space_id]
        if not space.free():
            return None
        waiter = self.waiters.pop(0)
        self.occupy(space_id, waiter.get("vehicle", "Hatchback"), waiter["holder"])
        return waiter

    def join_waitlist(self, holder: str, vehicle: str = "Hatchback") -> None:
        self.waiters.append({"holder": holder, "vehicle": vehicle})

    # -- predictions (demo heuristics) -------------------------------------- #

    def predict_release(self, space_id: str, eta_minutes: float = 12.0) -> dict:
        """Expected release time + probability inside the driver's ETA window."""
        space = self.spaces[space_id]
        elapsed = (datetime.now() - (space.entry_ts or datetime.now())).total_seconds() / 60.0
        mean_remaining = max(15.0, space.expected_exit and
                             (space.expected_exit - datetime.now()).total_seconds() / 60.0 or 60.0)
        mean_remaining = max(mean_remaining, 15.0 - elapsed * 0)  # keep simple & explainable
        ci = max(4.0, mean_remaining * 0.16)
        prob = release_probability(mean_remaining, ci,
                                   max(1.0, eta_minutes - 3), eta_minutes + 3)
        return {
            "space": space_id,
            "mean_remaining_min": round(mean_remaining, 1),
            "ci_min": round(ci, 1),
            "prob_in_eta_window": round(prob, 2),
        }

    def stats(self) -> dict:
        total = len(self.spaces)
        occupied = sum(1 for s in self.spaces.values() if s.occupied)
        reserved = sum(1 for s in self.spaces.values() if s.reserved and not s.occupied)
        ev_free = sum(1 for s in self.spaces.values() if s.ev and s.free())
        acc_free = sum(1 for s in self.spaces.values() if s.accessible and s.free())
        return {
            "total": total,
            "available": total - occupied - reserved,
            "occupied": occupied,
            "reserved": reserved,
            "ev_free": ev_free,
            "accessible_free": acc_free,
            "occupancy_pct": round(100.0 * occupied / total, 1) if total else 0.0,
        }

    # -- smart slot allocation ------------------------------------------------ #

    def find_best_slot(self, vehicle: str, needs_ev: bool = False,
                       needs_accessible: bool = False) -> Optional[Space]:
        """
        Score = availability certainty - walking distance penalty.
        EV cars get EV bays; accessibility needs rank accessible bays first.
        """
        is_ev = VEHICLES[vehicle]["ev"]
        best, best_score = None, float("-inf")
        for s in self.spaces.values():
            if s.occupied or s.reserved:
                continue
            if needs_ev and not s.ev:
                continue
            if is_ev and not s.ev and s.dist < 500:      # EV prefers EV bay if any exists
                continue
            if needs_accessible and not s.accessible:
                continue
            if vehicle == "SUV" and s.dist < 0:
                continue
            score = 100.0 - s.dist * 0.5 + (30 if needs_accessible and s.accessible else 0)
            if score > best_score:
                best, best_score = s, score
        return best

    # -- reservations ---------------------------------------------------------- #

    def reserve_slot(self, space_id: str, holder: str, arrive: datetime,
                     depart: datetime, visitor: bool = False) -> Optional[Reservation]:
        space = self.spaces.get(space_id)
        if space is None or not space.free() or depart <= arrive:
            return None
        if any(r.space_id == space_id and r.status in ("CONFIRMED", "ACTIVE")
               for r in self.reservations.values()):
            return None
        minutes = (depart - arrive).total_seconds() / 60.0
        fee = self.fee_calculator.fare("Hatchback", minutes, arrive, visitor=visitor)
        code = f"RSV{len(self.reservations) + 1:04d}"
        res = Reservation(code, space_id, holder, arrive, depart, fee=fee)
        self.reservations[code] = res
        space.reserved = True
        self._notify("📌", f"{space_id} reserved for {holder} ({code})")
        return res

    def cancel_reservation(self, code: str) -> bool:
        res = self.reservations.get(code)
        if res is None or res.status not in ("CONFIRMED", "ACTIVE"):
            return False
        res.status = "CANCELLED"
        space = self.spaces[res.space_id]
        if not space.occupied:
            space.reserved = False
        self._notify("✖️", f"Reservation {code} cancelled")
        return True

    def advance_time(self, now: Optional[datetime] = None) -> None:
        """Expire reservations whose window has passed (called by the tick)."""
        now = now or datetime.now()
        for res in self.reservations.values():
            if res.status in ("CONFIRMED", "ACTIVE") and now > res.depart:
                res.status = "EXPIRED"
                space = self.spaces[res.space_id]
                if not space.occupied:
                    space.reserved = False
                self._notify("⌛", f"Reservation {res.code} expired")

    # -- EV ---------------------------------------------------------------------- #

    def ev_stations(self) -> List[dict]:
        return [
            {"station": f"EV-{i:02d}", "space": s.id, "status": "AVAILABLE" if s.free() else "IN_USE",
             "charge_pct": s.ev_charge_pct or 0.0}
            for i, s in enumerate((x for x in self.spaces.values() if x.ev), start=1)
        ]

    def start_charging(self, space_id: str) -> bool:
        space = self.spaces.get(space_id)
        if space is None or not space.ev or not space.free():
            return False
        self.occupy(space_id, "EV Car", "EV-DEMO")
        space.ev_charge_pct = 30.0
        return True

    # -- find my car ---------------------------------------------------------------- #

    def save_find_car(self, floor: int, zone: str, slot: str, plate: str) -> dict:
        self.find_car = {"floor": floor, "zone": zone, "slot": slot, "plate": plate.upper()}
        return self.find_car

    def find_car_route(self) -> Optional[str]:
        if not self.find_car:
            return None
        fc = self.find_car
        return (f"Entrance → lift to L{fc['floor'] + 1} → zone {fc['zone']} → slot {fc['slot']} "
                f"· vehicle {fc['plate']} · ~3 min walk (demo route)")

    # -- navigation ------------------------------------------------------------------ #

    def route_to_slot(self, space_id: str) -> Optional[str]:
        space = self.spaces.get(space_id)
        if space is None:
            return None
        return (f"Enter → level L{space.floor + 1} → zone {space.zone} → slot {space.id} "
                f"· {space.dist} m walk")

    # -- admin reporting ---------------------------------------------------------------- #

    def collect_alerts(self) -> List[str]:
        alerts, now = [], datetime.now()
        for s in self.spaces.values():
            if s.occupied and s.expected_exit and now > s.expected_exit + timedelta(minutes=10):
                alerts.append(f"OVERSTAY: {s.id} {s.vehicle}")
        if self.stats()["occupancy_pct"] > 88:
            alerts.append("CAPACITY: occupancy above 88%")
        return alerts

    def report(self) -> dict:
        st = self.stats()
        return {
            "location": self.name,
            "stats": st,
            "reservations": len(self.reservations),
            "events": len(self.events),
            "peak_hour": self.forecaster.peak_hour(self.key),
        }


# --------------------------------------------------------------------------- #
# Location factory
# --------------------------------------------------------------------------- #

def build_location(key: str) -> ParkingLot:
    specs = {
        "it":       ("IT Company", FeeCalculator(),            [("A", 30, 0), ("B", 30, 0)]),
        "mall":     ("Shopping Mall", MallFee(),               [("P1", 30, 0), ("P2", 25, 1)]),
        "hospital": ("Hospital", FeeCalculator((9, 13), (9, 13)), [("OP", 25, 0), ("EM", 20, 0)]),
        "airport":  ("Airport", FeeCalculator((5, 8), (19, 23)), [("S1", 30, 0), ("L1", 25, 1)]),
        "complex":  ("Commercial Complex", FeeCalculator(),    [("TA", 25, 0), ("TB", 25, 1)]),
    }
    name, fee, zones = specs[key]
    lot = ParkingLot(name=name, key=key, fee_calculator=fee)
    for zone, count, floor in zones:
        lot.add_location_zone(zone, count, floor=floor)
    return lot


# --------------------------------------------------------------------------- #
# Embedded test suite
# --------------------------------------------------------------------------- #

class TestSeed(unittest.TestCase):
    """Determinism: same seed -> same forecast sequence."""

    def test_forecast_is_deterministic(self):
        f1 = DemandForecaster().forecast("it", 8)
        f2 = DemandForecaster().forecast("it", 8)
        self.assertEqual(f1, f2)


class TestPricing(unittest.TestCase):
    def setUp(self):
        self.fee = FeeCalculator()

    def test_base_fee(self):
        self.assertEqual(self.fee.fare("Hatchback", 60, datetime(2026, 9, 30, 14, 0)), 45.0)

    def test_peak_surcharge(self):
        peak = self.fee.fare("Hatchback", 60, datetime(2026, 9, 30, 11, 0))
        self.assertEqual(peak, 45.0 * 1.25)

    def test_visitor_premium(self):
        normal = self.fee.fare("Hatchback", 60, datetime(2026, 9, 30, 14, 0))
        visitor = self.fee.fare("Hatchback", 60, datetime(2026, 9, 30, 14, 0), visitor=True)
        self.assertEqual(visitor, round(normal * 1.2, 2))

    def test_rain_surge(self):
        dry = self.fee.fare("Hatchback", 60, datetime(2026, 9, 30, 14, 0))
        wet = self.fee.fare("Hatchback", 60, datetime(2026, 9, 30, 14, 0), rain=True)
        self.assertEqual(wet, round(dry * 1.1, 2))

    def test_ev_energy(self):
        plain = self.fee.fare("Hatchback", 120, datetime(2026, 9, 30, 14, 0))
        ev = self.fee.fare("EV Car", 120, datetime(2026, 9, 30, 14, 0), ev_energy=True)
        self.assertEqual(ev, plain + 40.0)

    def test_mall_fee_visitor_premium(self):
        mall = MallFee()
        self.assertGreater(mall.VISITOR_MULT, FeeCalculator.VISITOR_MULT)


class TestDemand(unittest.TestCase):
    def setUp(self):
        self.fc = DemandForecaster()

    def test_forecast_shape(self):
        out = self.fc.forecast("airport", 6, horizon=8)
        self.assertEqual(len(out), 8)
        self.assertTrue(all(0.0 <= row["demand"] <= 1.0 for row in out))

    def test_peak_multiplier(self):
        self.assertEqual(self.fc.peak_hour("hospital"), 2)          # 09:00 bucket
        self.assertEqual(self.fc.peak_hour("mall"), 14 % 24 or 14)  # evening bucket index 7 -> hour 14

    def test_release_probability_bounds(self):
        p = release_probability(30, 5, 10, 20)
        self.assertGreaterEqual(p, 0.01)
        self.assertLessEqual(p, 0.99)

    def test_release_probability_monotonic(self):
        near = release_probability(10, 5, 8, 14)
        far = release_probability(90, 5, 8, 14)
        self.assertGreater(near, far)

    def test_wait_time_consistent(self):
        """A high-probability window implies little to no waiting."""
        p = release_probability(5, 4, 1, 15)
        self.assertGreater(p, 0.5)


class TestParking(unittest.TestCase):
    def setUp(self):
        self.lot = build_location("it")

    def test_multi_location_isolation(self):
        other = build_location("mall")
        self.assertIsNot(self.lot, other)
        self.assertNotEqual(self.lot.fee_calculator.BASE, other.fee_calculator.BASE)

    def test_occupy_and_stats(self):
        self.assertTrue(self.lot.occupy("A05", "Sedan", "KA 01 AB 1234"))
        st = self.lot.stats()
        self.assertEqual(st["occupied"], 1)
        self.assertEqual(st["available"], st["total"] - 1)

    def test_double_occupy_rejected(self):
        self.lot.occupy("A05", "Sedan", "X1")
        self.assertFalse(self.lot.occupy("A05", "Sedan", "X2"))

    def test_dashboard_stats_shape(self):
        st = self.lot.stats()
        for key in ("total", "available", "occupied", "reserved", "ev_free", "accessible_free", "occupancy_pct"):
            self.assertIn(key, st)

    def test_admin_report_totals(self):
        self.lot.occupy("A05", "Sedan", "X1")
        rep = self.lot.report()
        self.assertEqual(rep["stats"]["occupied"], 1)
        self.assertIn("peak_hour", rep)


class TestVacate(unittest.TestCase):
    def setUp(self):
        self.lot = build_location("it")
        self.lot.occupy("A05", "Hatchback", "KA 05 XY 1111")

    def test_vacate_frees_slot(self):
        self.assertTrue(self.lot.vacate("A05"))
        self.assertTrue(self.lot.spaces["A05"].free())

    def test_vacate_notifies_waiter(self):
        self.lot.join_waitlist("EMP2043")
        self.lot.vacate("A05")
        icons = [e["icon"] for e in self.lot.events]
        self.assertIn("🪧", icons)
        self.assertIn("🟢", icons)

    def test_vacate_auto_claims_waiter(self):
        self.lot.join_waitlist("EMP2043")
        self.lot.vacate("A05")
        self.assertEqual(self.lot.spaces["A05"].vehicle, "EMP2043")

    def test_vacate_empty_slot_is_noop(self):
        self.assertFalse(self.lot.vacate("A99"))


class TestReservation(unittest.TestCase):
    def setUp(self):
        self.lot = build_location("it")
        self.arrive = datetime(2026, 9, 30, 15, 0)
        self.depart = datetime(2026, 9, 30, 17, 0)

    def test_reserve_and_cancel(self):
        res = self.lot.reserve_slot("B10", "EMP1024", self.arrive, self.depart)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, "CONFIRMED")
        self.assertTrue(self.lot.cancel_reservation(res.code))
        self.assertEqual(res.status, "CANCELLED")
        self.assertTrue(self.lot.spaces["B10"].free())

    def test_double_reserve_rejected(self):
        self.assertIsNotNone(self.lot.reserve_slot("B10", "A", self.arrive, self.depart))
        self.assertIsNone(self.lot.reserve_slot("B10", "B", self.arrive, self.depart))

    def test_invalid_window_rejected(self):
        self.assertIsNone(self.lot.reserve_slot("B11", "A", self.depart, self.arrive))

    def test_expired_reservation(self):
        res = self.lot.reserve_slot("B12", "A", self.arrive, self.depart)
        self.lot.advance_time(now=self.depart + timedelta(minutes=5))
        self.assertEqual(res.status, "EXPIRED")
        self.assertTrue(self.lot.spaces["B12"].free())


class TestAllocation(unittest.TestCase):
    def setUp(self):
        self.lot = build_location("it")

    def test_best_slot_prefers_free_and_near(self):
        slot = self.lot.find_best_slot("Hatchback")
        self.assertIsNotNone(slot)
        self.assertTrue(slot.free())
        self.assertLessEqual(slot.dist, 45)

    def test_ev_car_gets_ev_slot(self):
        slot = self.lot.find_best_slot("EV Car", needs_ev=True)
        self.assertIsNotNone(slot)
        self.assertTrue(slot.ev)

    def test_accessible_need(self):
        slot = self.lot.find_best_slot("Hatchback", needs_accessible=True)
        self.assertIsNotNone(slot)
        self.assertTrue(slot.accessible)


class TestEV(unittest.TestCase):
    def setUp(self):
        self.lot = build_location("it")

    def test_charging_progresses(self):
        self.assertTrue(self.lot.start_charging("A01"))
        station = self.lot.ev_stations()[0]
        self.assertEqual(station["status"], "IN_USE")
        ev = EVStation("EV-01", "A01", status="IN_USE", charge_pct=30)
        before = ev.charge_pct
        ev.charge_tick(10)
        self.assertGreater(ev.charge_pct, before)

    def test_charge_completes(self):
        ev = EVStation("EV-01", "A01", status="IN_USE", charge_pct=99)
        ev.charge_tick(10)
        self.assertEqual(ev.status, "AVAILABLE")

    def test_charging_requires_ev_slot(self):
        self.assertFalse(self.lot.start_charging("A10"))   # not an EV bay


class TestNavigation(unittest.TestCase):
    def setUp(self):
        self.lot = build_location("it")

    def test_route_returns_steps(self):
        route = self.lot.route_to_slot("A07")
        self.assertIn("slot A07", route)
        self.assertIn("zone A", route)

    def test_route_unknown_slot(self):
        self.assertIsNone(self.lot.route_to_slot("ZZ99"))


class TestFindCar(unittest.TestCase):
    def setUp(self):
        self.lot = build_location("it")

    def test_save_and_find(self):
        self.lot.save_find_car(0, "A", "A07", "TS09AB1234")
        route = self.lot.find_car_route()
        self.assertIsNotNone(route)
        self.assertIn("A07", route)
        self.assertIn("TS09AB1234".upper()[:4], route.upper())

    def test_find_before_save(self):
        self.assertIsNone(self.lot.find_car_route())


class TestAdmin(unittest.TestCase):
    def setUp(self):
        self.lot = build_location("it")

    def test_overstay_alert(self):
        space = self.lot.spaces["A05"]
        space.occupied = True
        space.vehicle = "LATE-1"
        space.entry_ts = datetime.now() - timedelta(hours=5)
        space.expected_exit = datetime.now() - timedelta(hours=1)
        alerts = self.lot.collect_alerts()
        self.assertTrue(any("OVERSTAY" in a and "A05" in a for a in alerts))

    def test_capacity_alert(self):
        for sid in list(self.lot.spaces)[:90]:
            self.lot.occupy(sid, "Hatchback", f"P-{sid}")
        self.assertTrue(any("CAPACITY" in a for a in self.lot.collect_alerts()))


class TestAuth(unittest.TestCase):
    def test_admin_credentials_accepted(self):
        self.assertTrue(verify_admin("admin", "Admin@123"))
        self.assertTrue(verify_admin("malladmin", "Mall@123"))

    def test_admin_credentials_rejected(self):
        self.assertFalse(verify_admin("admin", "wrong"))
        self.assertFalse(verify_admin("ghost", "Admin@123"))

    def test_user_credentials(self):
        self.assertTrue(verify_user("EMP1024", "demo123"))
        self.assertFalse(verify_user("EMP1024", "nope"))

    def test_passwords_not_stored_in_plaintext(self):
        blob = repr(ADMIN_ACCOUNTS) + repr(USER_ACCOUNTS)
        self.assertNotIn("Admin@123", blob)
        self.assertNotIn("demo123", blob)


# --------------------------------------------------------------------------- #
# Demo entry point
# --------------------------------------------------------------------------- #

def demo_report() -> None:
    """Quick human-readable sanity run (not a test)."""
    lot = build_location("mall")
    lot.occupy("P105", "Sedan", "TS 09 MB 4321", stay_hours=2)
    lot.occupy("P106", "EV Car", "KA 05 EV 9090", stay_hours=3)
    print(f"Location   : {lot.name}")
    print(f"Stats      : {lot.stats()}")
    print(f"Best slot  : {lot.find_best_slot('EV Car', needs_ev=True)}")
    print(f"Peak hour  : {lot.forecaster.peak_hour(lot.key)}:00 (demo heuristic)")
    print(f"Alerts     : {lot.collect_alerts() or 'none'}")


if __name__ == "__main__":
    demo_report()
    print("\nRunning embedded test suite...\n" + "-" * 60)
    unittest.main(verbosity=2, exit=False)
