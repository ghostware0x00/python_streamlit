from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ZONES: dict[str, int] = {
    "Central Library - Main Hall": 180,
    "PG & MCA Quiet Study Zone": 90,
    "Knowledge Commons & Digital Lab": 75,
    "Collaborative Discussion Pods": 48,
}

BOOKS_CATALOG: list[dict[str, object]] = [
    {"book_id": "BK-101", "title": "Operating System Concepts", "department": "MCA / CS", "total_copies": 12},
    {"book_id": "BK-102", "title": "Database System Concepts", "department": "MCA / CS", "total_copies": 10},
    {"book_id": "BK-103", "title": "Computer Networks", "department": "MCA / CS", "total_copies": 8},
    {"book_id": "BK-104", "title": "Python for Data Analysis", "department": "MCA / Data", "total_copies": 9},
    {"book_id": "BK-105", "title": "Introduction to Algorithms", "department": "MCA / CS", "total_copies": 7},
    {"book_id": "BK-106", "title": "Research Methodology", "department": "PG Research", "total_copies": 14},
    {"book_id": "BK-107", "title": "Software Engineering", "department": "MCA / CS", "total_copies": 11},
    {"book_id": "BK-108", "title": "Artificial Intelligence: A Modern Approach", "department": "MCA / AI", "total_copies": 6},
]


def _hour_demand_curve(hour: int, exam_intensity: float) -> float:
    """Computes hourly campus demand multiplier scaled by exam/peak intensity."""
    morning_peak = np.exp(-0.5 * ((hour - 11.0) / 2.0) ** 2)
    afternoon_peak = np.exp(-0.5 * ((hour - 16.0) / 2.5) ** 2)
    base = 0.12 + 0.55 * morning_peak + 0.65 * afternoon_peak
    if hour < 8 or hour > 21:
        base *= 0.25
    scaled = base * (0.75 + 0.45 * exam_intensity) + (0.08 * exam_intensity)
    return float(np.clip(scaled, 0.10, 1.0))


def generate_seat_history(days: int = 14, exam_intensity: float = 0.85, seed: int = 42) -> pd.DataFrame:
    """Generates hourly seat occupancy telemetry across library zones."""
    rng = np.random.default_rng(seed)
    end_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)
    timestamps = pd.date_range(start=start_dt, end=end_dt, freq="h")

    rows: list[dict[str, object]] = []
    for ts in timestamps:
        weekend_factor = 0.65 if ts.dayofweek == 6 else (0.85 if ts.dayofweek == 5 else 1.0)
        curve = _hour_demand_curve(ts.hour, exam_intensity) * weekend_factor

        for zone, capacity in ZONES.items():
            zone_bias = {
                "Central Library - Main Hall": 1.02,
                "PG & MCA Quiet Study Zone": 1.08 if exam_intensity >= 0.7 else 0.90,
                "Knowledge Commons & Digital Lab": 0.95,
                "Collaborative Discussion Pods": 0.88,
            }.get(zone, 1.0)

            occ_ratio = float(np.clip(curve * zone_bias + rng.normal(0, 0.05), 0.05, 0.99))
            occupied = int(round(occ_ratio * capacity))
            occupied = max(0, min(capacity, occupied))
            avg_dwell_mins = int(np.clip(rng.normal(75 + 45 * exam_intensity, 20), 25, 240))

            rows.append(
                {
                    "timestamp": ts,
                    "zone": zone,
                    "capacity": capacity,
                    "occupied_seats": occupied,
                    "available_seats": capacity - occupied,
                    "occupancy_pct": round((occupied / capacity) * 100, 1),
                    "avg_dwell_mins": avg_dwell_mins,
                }
            )

    return pd.DataFrame(rows)


def _classify_book(pressure: float, available_copies: int) -> str:
    if pressure >= 130 or available_copies == 0:
        return "Add Copies / Short Loan"
    if pressure >= 95:
        return "Monitor Waitlist"
    return "Sufficient Stock"


def generate_book_demand(exam_intensity: float = 0.85, seed: int = 42) -> pd.DataFrame:
    """Generates book checkout, active hold/waitlist, and demand pressure metrics."""
    rng = np.random.default_rng(seed + 10)
    records: list[dict[str, object]] = []
    mult = 0.80 + 0.55 * exam_intensity

    for item in BOOKS_CATALOG:
        total_copies = int(item["total_copies"])
        issued = int(np.clip(round(total_copies * rng.uniform(0.45, 0.92) * mult), 1, total_copies))
        available_copies = max(0, total_copies - issued)
        scarcity_bonus = max(0, issued - int(total_copies * 0.7)) * rng.integers(1, 3)
        active_holds = int(max(0, rng.poisson(lam=2.2 * mult) + scarcity_bonus))
        weekly_checkouts = int(max(issued, rng.poisson(lam=total_copies * 2.3 * mult)))
        demand_pressure = round(((issued + active_holds) / max(1, total_copies)) * 100, 1)

        records.append(
            {
                "book_id": item["book_id"],
                "title": item["title"],
                "department": item["department"],
                "total_copies": total_copies,
                "issued_copies": issued,
                "available_copies": available_copies,
                "active_holds": active_holds,
                "weekly_checkouts": weekly_checkouts,
                "demand_pressure_pct": demand_pressure,
                "recommended_action": _classify_book(demand_pressure, available_copies),
            }
        )

    return pd.DataFrame(records).sort_values("demand_pressure_pct", ascending=False).reset_index(drop=True)


def simulate_live_event(
    seat_df: pd.DataFrame,
    book_df: pd.DataFrame,
    step: int,
    mode: str = "Random",
    intensity_high: bool = True,
    preferred_zones: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Applies one real-time event and returns updated frames plus structured event metadata."""
    rng = np.random.default_rng(1000 + step)
    now_ts = datetime.now().replace(second=0, microsecond=0) + timedelta(seconds=step * 15)
    updated_seats = seat_df.copy()
    updated_books = book_df.copy()

    if mode == "Seat Only":
        event_type = str(rng.choice(["seat_checkin", "seat_checkout"], p=[0.6, 0.4]))
    elif mode == "Book Only":
        event_type = str(rng.choice(["book_hold", "book_return"], p=[0.65, 0.35]))
    else:
        event_type = str(
            rng.choice(["seat_checkin", "seat_checkout", "book_hold", "book_return"], p=[0.35, 0.25, 0.25, 0.15])
        )

    step_min, step_max = (6, 16) if intensity_high else (1, 4)

    if event_type in ("seat_checkin", "seat_checkout"):
        zone_pool = preferred_zones if preferred_zones else list(ZONES.keys())
        zone = str(rng.choice(zone_pool))
        mask = updated_seats["zone"] == zone
        latest_idx = updated_seats[mask].index[-1]
        cap = int(updated_seats.loc[latest_idx, "capacity"])
        curr_occ = int(updated_seats.loc[latest_idx, "occupied_seats"])
        delta = int(rng.integers(step_min, step_max + 1))

        if event_type == "seat_checkin":
            new_occ = min(cap, curr_occ + delta)
            actual_delta = new_occ - curr_occ
            summary = f"+{actual_delta} seat check-ins in {zone}"
        else:
            new_occ = max(0, curr_occ - delta)
            actual_delta = -(curr_occ - new_occ)
            summary = f"{actual_delta} seat check-outs in {zone}"

        new_pct = round((new_occ / cap) * 100, 1)
        new_row = updated_seats.loc[latest_idx].copy()
        new_row["timestamp"] = now_ts
        new_row["occupied_seats"] = new_occ
        new_row["available_seats"] = cap - new_occ
        new_row["occupancy_pct"] = new_pct
        updated_seats = pd.concat([updated_seats, pd.DataFrame([new_row])], ignore_index=True)

        meta: dict[str, object] = {
            "category": "seat",
            "time": now_ts.strftime("%H:%M:%S"),
            "target": zone,
            "summary": summary,
            "detail": f"Occupancy now {new_occ}/{cap} seats ({new_pct}%)",
        }
    else:
        idx = int(rng.integers(0, len(updated_books)))
        title = str(updated_books.loc[idx, "title"])
        total = int(updated_books.loc[idx, "total_copies"])
        issued = int(updated_books.loc[idx, "issued_copies"])
        holds = int(updated_books.loc[idx, "active_holds"])
        delta = int(rng.integers(1, 4 if intensity_high else 2))

        if event_type == "book_hold":
            holds += delta
            newly_issued = min(total - issued, int(rng.integers(0, delta + 1)))
            issued += newly_issued
            summary = f"+{delta} hold(s) on '{title}'"
        else:
            returned = min(issued, delta)
            issued -= returned
            fulfilled_holds = min(holds, max(1, returned))
            holds -= fulfilled_holds
            summary = f"{returned} copy return(s) for '{title}'"

        avail = max(0, total - issued)
        pressure = round(((issued + holds) / max(1, total)) * 100, 1)
        updated_books.loc[idx, "issued_copies"] = issued
        updated_books.loc[idx, "active_holds"] = holds
        updated_books.loc[idx, "available_copies"] = avail
        updated_books.loc[idx, "demand_pressure_pct"] = pressure
        updated_books.loc[idx, "recommended_action"] = _classify_book(pressure, avail)
        updated_books = updated_books.sort_values("demand_pressure_pct", ascending=False).reset_index(drop=True)

        meta = {
            "category": "book",
            "time": now_ts.strftime("%H:%M:%S"),
            "target": title,
            "summary": summary,
            "detail": f"Available: {avail}/{total} | Holds: {holds} | Pressure: {pressure}%",
        }

    return updated_seats, updated_books, meta


def load_custom_seat_csv(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """Validates and normalizes a user-uploaded seat occupancy CSV."""
    required = {"timestamp", "zone", "capacity", "occupied_seats"}
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, f"Could not parse CSV: {exc}"

    missing = required - set(df.columns)
    if missing:
        return None, f"Missing required columns: {', '.join(sorted(missing))}"

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["occupied_seats"] = pd.to_numeric(df["occupied_seats"], errors="coerce")
    df = df.dropna(subset=["timestamp", "capacity", "occupied_seats"])
    df = df[df["capacity"] > 0]
    df["occupied_seats"] = df["occupied_seats"].clip(lower=0).astype(int)
    df["capacity"] = df["capacity"].astype(int)
    df["available_seats"] = (df["capacity"] - df["occupied_seats"]).clip(lower=0)
    df["occupancy_pct"] = ((df["occupied_seats"] / df["capacity"]) * 100).round(1)
    if "avg_dwell_mins" not in df.columns:
        df["avg_dwell_mins"] = 90
    return df.sort_values("timestamp").reset_index(drop=True), None