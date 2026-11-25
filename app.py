from __future__ import annotations

from datetime import timedelta
import time
import pandas as pd
import plotly.express as px
import streamlit as st

from data_engine import (
    BOOKS_CATALOG,
    ZONES,
    generate_book_demand,
    generate_seat_history,
    load_custom_seat_csv,
    simulate_live_event,
)

st.set_page_config(
    page_title="CHRIST Library Seat & Book Demand Analytics",
    page_icon="📚",
    layout="wide",
)

ALL_DEPARTMENTS = sorted({str(b["department"]) for b in BOOKS_CATALOG})


def init_state() -> None:
    if "exam_intensity" not in st.session_state:
        st.session_state.exam_intensity = 0.85
    if "seat_df" not in st.session_state:
        st.session_state.seat_df = generate_seat_history(days=14, exam_intensity=st.session_state.exam_intensity)
    if "book_df" not in st.session_state:
        st.session_state.book_df = generate_book_demand(exam_intensity=st.session_state.exam_intensity)
    if "sim_step" not in st.session_state:
        st.session_state.sim_step = 0
    if "last_event" not in st.session_state:
        st.session_state.last_event = None
    if "event_log" not in st.session_state:
        st.session_state.event_log = []


def apply_events(count: int, mode: str, intensity_high: bool, preferred_zones: list[str]) -> None:
    for _ in range(count):
        st.session_state.sim_step += 1
        s_df, b_df, meta = simulate_live_event(
            st.session_state.seat_df,
            st.session_state.book_df,
            st.session_state.sim_step,
            mode=mode,
            intensity_high=intensity_high,
            preferred_zones=preferred_zones,
        )
        st.session_state.seat_df = s_df
        st.session_state.book_df = b_df
        st.session_state.last_event = meta
        st.session_state.event_log.insert(0, f"[{meta['time']}] {meta['summary']} ({meta['detail']})")
    st.session_state.event_log = st.session_state.event_log[:25]


def main() -> None:
    init_state()

    st.title("Library Seat & Book Demand Analytics Dashboard")
    st.caption("CHRIST University Smart Campus Analytics • Real-Time Seat Occupancy & Book Demand Intelligence")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Simulation & Filters")

        intensity = st.slider(
            "Academic Demand Load (CIA / Exam Surge)",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.exam_intensity,
            step=0.05,
        )
        if abs(intensity - st.session_state.exam_intensity) > 1e-6:
            st.session_state.exam_intensity = intensity
            st.session_state.seat_df = generate_seat_history(days=14, exam_intensity=intensity)
            st.session_state.book_df = generate_book_demand(exam_intensity=intensity)
            st.session_state.sim_step = 0
            st.session_state.last_event = None
            st.session_state.event_log.insert(0, f"Regenerated baseline at {int(intensity * 100)}% demand load.")

        selected_zones = st.multiselect(
            "Library Zones",
            options=list(ZONES.keys()),
            default=list(ZONES.keys()),
        )
        if not selected_zones:
            selected_zones = list(ZONES.keys())

        selected_depts = st.multiselect(
            "Book Departments",
            options=ALL_DEPARTMENTS,
            default=ALL_DEPARTMENTS,
        )
        if not selected_depts:
            selected_depts = ALL_DEPARTMENTS

        st.divider()
        st.subheader("Real-Time Event Simulator")
        sim_mode = st.selectbox("Event Type Filter", options=["Random", "Seat Only", "Book Only"])
        high_impact = st.toggle("High-Impact Step Size (6–16 seats)", value=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Simulate Event", type="primary", use_container_width=True):
                apply_events(1, sim_mode, high_impact, selected_zones)
        with b2:
            if st.button("Burst (+5 Events)", use_container_width=True):
                apply_events(5, sim_mode, high_impact, selected_zones)

        c1, c2 = st.columns(2)
        with c1:
            auto_stream = st.checkbox("Auto-Stream (2s)", value=False)
        with c2:
            if st.button("Reset Data", use_container_width=True):
                st.session_state.seat_df = generate_seat_history(days=14, exam_intensity=st.session_state.exam_intensity)
                st.session_state.book_df = generate_book_demand(exam_intensity=st.session_state.exam_intensity)
                st.session_state.sim_step = 0
                st.session_state.last_event = None
                st.session_state.event_log = []

        st.divider()
        st.subheader("Custom Seat CSV")
        uploaded = st.file_uploader("Upload seat telemetry CSV", type=["csv"])
        if uploaded is not None:
            custom_df, err = load_custom_seat_csv(uploaded)
            if err:
                st.error(err)
            elif custom_df is not None:
                st.session_state.seat_df = custom_df
                st.success("Custom seat CSV loaded.")

    # --- Live Event Spotlight Banner ---
    if st.session_state.last_event:
        ev = st.session_state.last_event
        st.success(f"**Latest Live Event ({ev['time']}):** {ev['summary']} — *{ev['detail']}*")
    else:
        st.info("Click **Simulate Event** or **Burst (+5 Events)** in the sidebar to trigger live seat or book activity.")

    # Filter data
    seat_filtered = st.session_state.seat_df[st.session_state.seat_df["zone"].isin(selected_zones)].copy()
    book_filtered = st.session_state.book_df[st.session_state.book_df["department"].isin(selected_depts)].copy()

    latest_per_zone = seat_filtered.sort_values("timestamp").groupby("zone", as_index=False).last()
    total_cap = int(latest_per_zone["capacity"].sum()) if not latest_per_zone.empty else 0
    total_occ = int(latest_per_zone["occupied_seats"].sum()) if not latest_per_zone.empty else 0
    total_avail = max(0, total_cap - total_occ)
    curr_occ_pct = round((total_occ / total_cap) * 100, 1) if total_cap > 0 else 0.0
    high_demand_books = int((book_filtered["demand_pressure_pct"] >= 100).sum())

    # --- Top KPI Strip ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Seat Occupancy", f"{curr_occ_pct}%", f"{total_occ} / {total_cap} seats")
    k2.metric("Available Seats Now", f"{total_avail}", "Open seats across zones")
    k3.metric("High-Demand Book Titles", f"{high_demand_books}", "Pressure >= 100%")
    k4.metric("Simulated Live Events", f"{st.session_state.sim_step}", "Total events applied")

    st.divider()

    # --- Section 1: Seat Analytics ---
    st.subheader("1. Real-Time Seat Occupancy & Congestion Analytics")
    window_choice = st.radio(
        "Time Window for Trend Chart",
        options=["Last 24 Hours (Live View)", "Full 14-Day History"],
        horizontal=True,
    )

    chart_seat_df = seat_filtered
    if window_choice.startswith("Last 24") and not seat_filtered.empty:
        max_ts = seat_filtered["timestamp"].max()
        chart_seat_df = seat_filtered[seat_filtered["timestamp"] >= (max_ts - timedelta(hours=24))]

    c_left, c_right = st.columns([1.4, 1.0])
    with c_left:
        if not chart_seat_df.empty:
            fig_trend = px.line(
                chart_seat_df,
                x="timestamp",
                y="occupancy_pct",
                color="zone",
                markers=window_choice.startswith("Last 24"),
                title="Seat Occupancy Rate Over Time (%)",
                labels={"timestamp": "Timestamp", "occupancy_pct": "Occupancy (%)", "zone": "Zone"},
            )
            fig_trend.update_layout(yaxis_range=[0, 105], height=360)
            st.plotly_chart(fig_trend, use_container_width=True)

    with c_right:
        if not latest_per_zone.empty:
            fig_bar = px.bar(
                latest_per_zone,
                x="zone",
                y=["occupied_seats", "available_seats"],
                title="Current Seat Allocation by Zone",
                barmode="stack",
                labels={"value": "Seats", "zone": "Zone", "variable": "Status"},
            )
            fig_bar.update_layout(height=360, xaxis_tickangle=-18)
            st.plotly_chart(fig_bar, use_container_width=True)

    if not seat_filtered.empty:
        hm_df = seat_filtered.copy()
        hm_df["day"] = pd.Categorical(
            hm_df["timestamp"].dt.day_name(),
            categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ordered=True,
        )
        hm_df["hour"] = hm_df["timestamp"].dt.hour
        pivot_hm = hm_df.pivot_table(index="day", columns="hour", values="occupancy_pct", aggfunc="mean", observed=False)
        fig_hm = px.imshow(
            pivot_hm,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Average Occupancy Heatmap (Day of Week vs. Hour of Day)",
            labels={"x": "Hour of Day (24h)", "y": "Day", "color": "Avg Occ %"},
        )
        fig_hm.update_layout(height=290)
        st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # --- Section 2: Book Demand Analytics ---
    st.subheader("2. Book Demand vs. Copy Supply Analytics")
    b_left, b_right = st.columns([1.25, 1.0])

    with b_left:
        if not book_filtered.empty:
            fig_demand = px.bar(
                book_filtered,
                x="demand_pressure_pct",
                y="title",
                color="recommended_action",
                orientation="h",
                title="Book Demand Pressure Index (%) — (Issued + Holds) / Total Copies",
                labels={"demand_pressure_pct": "Demand Pressure (%)", "title": "Book Title"},
            )
            fig_demand.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="100% Demand Threshold")
            fig_demand.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_demand, use_container_width=True)

    with b_right:
        st.markdown("**Circulation & Restock Priority Table**")
        st.dataframe(
            book_filtered[
                ["title", "department", "available_copies", "active_holds", "demand_pressure_pct", "recommended_action"]
            ],
            use_container_width=True,
            hide_index=True,
            height=340,
        )

    st.divider()

    # --- Section 3: Live Log, Recommendations & Export ---
    l_col, r_col = st.columns([1.2, 1.0])

    with l_col:
        st.subheader("Live Event Log")
        if st.session_state.event_log:
            for line in st.session_state.event_log[:10]:
                st.write(f"• {line}")
        else:
            st.caption("No live events simulated yet.")

    with r_col:
        st.subheader("Actionable Recommendations & Export")
        if not latest_per_zone.empty:
            congested = latest_per_zone[latest_per_zone["occupancy_pct"] >= 85]["zone"].tolist()
            if congested:
                st.warning(f"High congestion (>=85%): {', '.join(congested)}. Route overflow students to quieter zones.")
            else:
                st.info("Selected zones are currently below the 85% congestion threshold.")

        urgent_books = book_filtered[book_filtered["recommended_action"] == "Add Copies / Short Loan"]["title"].tolist()
        if urgent_books:
            st.error(f"Urgent restock / 2-hr short loan: {', '.join(urgent_books[:4])}.")
        else:
            st.success("Book stock levels are currently within normal thresholds.")

        st.download_button(
            label="Download Book Demand CSV",
            data=book_filtered.to_csv(index=False).encode("utf-8"),
            file_name="library_book_demand_live.csv",
            mime="text/csv",
        )

    if auto_stream:
        time.sleep(2)
        apply_events(1, sim_mode, high_impact, selected_zones)
        st.rerun()


if __name__ == "__main__":
    main()
