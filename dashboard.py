"""
================================================================
  CNC Machine – Edge-Based Early Failure Warning System
  Streamlit Dashboard  ·  dashboard.py
================================================================
  Run with:
      streamlit run dashboard.py

  Requires cnc_health_monitor.py in the SAME folder.
  Install deps:
      pip install streamlit pandas
================================================================
"""

import time
import pandas as pd
import streamlit as st

# ── Import core logic from your existing project file ────────
from cnc_health_monitor import (
    generate_sensor_data,
    detect_anomalies,
    calculate_health_score,
    get_suggestions,
    VIBRATION_THRESHOLD,
    TEMPERATURE_THRESHOLD,
    PRESSURE_THRESHOLD,
    SOUND_THRESHOLD,
    UPDATE_INTERVAL_SEC,
)

HISTORY_SIZE = 30   # rolling window for charts

# ================================================================
#  PAGE CONFIG  – must be the very first Streamlit call
# ================================================================
st.set_page_config(
    page_title="CNC Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
#  SESSION STATE
# ================================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "cycle" not in st.session_state:
    st.session_state.cycle = 0

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["Cycle", "Vibration", "Temperature", "Pressure", "Sound", "Health"]
    )

# Snapshot of the latest reading – persists across reruns so we can
# render the UI even while waiting for the next data tick.
if "latest" not in st.session_state:
    st.session_state.latest = None

# Chart placeholder – created ONCE and reused every rerun so the
# "System Trends" section never duplicates in the DOM.
if "chart_placeholder" not in st.session_state:
    st.session_state.chart_placeholder = None

# ================================================================
#  SIDEBAR
# ================================================================
with st.sidebar:
    st.title("⚙ System Config")
    st.markdown("---")

    st.markdown("**Thresholds**")
    st.markdown(f"- 🔵 Vibration: **{VIBRATION_THRESHOLD} mm/s**")
    st.markdown(f"- 🔴 Temperature: **{TEMPERATURE_THRESHOLD} °C**")
    st.markdown(f"- 🟠 Pressure: **{PRESSURE_THRESHOLD} bar**")
    st.markdown(f"- 🟣 Sound: **{SOUND_THRESHOLD} dB**")

    st.markdown("---")
    st.markdown("**Score Legend**")
    st.markdown("🟢 90–100 → Healthy")
    st.markdown("🟡 70–89  → Warning")
    st.markdown("🔴 0–69   → Critical")

    st.markdown("---")
    st.markdown(f"**Update Interval:** {UPDATE_INTERVAL_SEC}s")
    st.markdown(f"**History Window:** Last {HISTORY_SIZE} readings")
    st.caption("CNC Health Monitor v2.1")

# ================================================================
#  HEADER
# ================================================================
st.title("⚙ CNC Machine Health Monitor")
st.markdown("---")

# ================================================================
#  START / STOP BUTTON
# ================================================================
col_btn, col_status = st.columns([2, 5])

with col_btn:
    if st.button("▶ Start Monitoring" if not st.session_state.running else "⏹ Stop Monitoring"):
        st.session_state.running = not st.session_state.running
        st.rerun()

with col_status:
    if st.session_state.running:
        st.success(f"● LIVE  ·  Cycle #{st.session_state.cycle}")
    else:
        st.warning("◼ Stopped — press Start to begin")

st.markdown("---")

# ================================================================
#  DATA COLLECTION  (happens BEFORE UI rendering)
# ================================================================
if st.session_state.running:
    st.session_state.cycle += 1

    data            = generate_sensor_data()
    anomalies       = detect_anomalies(data)
    score, status   = calculate_health_score(data, anomalies)
    suggestions     = get_suggestions(anomalies)

    # Save snapshot so we can render without re-calling the functions
    st.session_state.latest = {
        "data": data, "anomalies": anomalies,
        "score": score, "status": status,
        "suggestions": suggestions,
    }

    # Append to rolling history
    new_row = pd.DataFrame([{
        "Cycle":       st.session_state.cycle,
        "Vibration":   data["vibration"],
        "Temperature": data["temperature"],
        "Pressure":    data["pressure"],
        "Sound":       data["sound"],
        "Health":      score,
    }])
    st.session_state.history = pd.concat(
        [st.session_state.history, new_row], ignore_index=True
    ).tail(HISTORY_SIZE)

# ================================================================
#  UI RENDER  (uses session_state.latest so it's always consistent)
# ================================================================
snap = st.session_state.latest

if snap is not None:
    data        = snap["data"]
    anomalies   = snap["anomalies"]
    score       = snap["score"]
    status      = snap["status"]
    suggestions = snap["suggestions"]

    # ── STATUS ──────────────────────────────────────────────
    st.subheader("Machine Status")
    if status == "Healthy":
        st.success(f"🟢 Healthy  |  Score: {score} / 100")
    elif status == "Warning":
        st.warning(f"🟡 Warning  |  Score: {score} / 100")
    else:
        st.error(f"🔴 Critical  |  Score: {score} / 100")

    # ── SENSOR METRICS ───────────────────────────────────────
    st.subheader("Sensor Readings")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🔵 Vibration",
        f"{data['vibration']} mm/s",
        delta="⚠ OVER" if anomalies["vibration"] else "OK",
        delta_color="inverse" if anomalies["vibration"] else "off",
    )
    c2.metric(
        "🔴 Temperature",
        f"{data['temperature']} °C",
        delta="⚠ OVER" if anomalies["temperature"] else "OK",
        delta_color="inverse" if anomalies["temperature"] else "off",
    )
    c3.metric(
        "🟠 Pressure",
        f"{data['pressure']} bar",
        delta="⚠ OVER" if anomalies["pressure"] else "OK",
        delta_color="inverse" if anomalies["pressure"] else "off",
    )
    c4.metric(
        "🟣 Sound",
        f"{data['sound']} dB",
        delta="⚠ OVER" if anomalies["sound"] else "OK",
        delta_color="inverse" if anomalies["sound"] else "off",
    )

    # ── MAINTENANCE SUGGESTIONS ──────────────────────────────
    st.subheader("Maintenance Suggestions")
    if suggestions:
        for s in suggestions:
            st.warning(f"🔧 {s}")
    else:
        st.success("✅ All systems normal — no action needed")

else:
    st.info("Press **▶ Start Monitoring** to begin collecting data.")

# ================================================================
#  CHARTS  – the placeholder is stamped into the page layout exactly
#  once (first run). Every subsequent rerun just overwrites its
#  contents, so "System Trends" can never appear twice.
# ================================================================
st.markdown("---")

# Stamp the placeholder into the page on the very first run only.
if st.session_state.chart_placeholder is None:
    st.session_state.chart_placeholder = st.empty()

chart_area = st.session_state.chart_placeholder
df = st.session_state.history

with chart_area.container():
    st.subheader("System Trends")

    if not df.empty:
        chart_df = df.set_index("Cycle")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Health Score", "Vibration", "Temperature", "Pressure", "Sound"
        ])

        with tab1:
            st.line_chart(chart_df[["Health"]], height=220, use_container_width=True)

        with tab2:
            st.line_chart(chart_df[["Vibration"]], height=220, use_container_width=True)

        with tab3:
            st.line_chart(chart_df[["Temperature"]], height=220, use_container_width=True)

        with tab4:
            st.line_chart(chart_df[["Pressure"]], height=220, use_container_width=True)

        with tab5:
            st.line_chart(chart_df[["Sound"]], height=220, use_container_width=True)

    else:
        st.info("Charts will appear after the first reading.")

# ================================================================
#  AUTO-REFRESH  – sleep then rerun ONLY if monitoring is active
#  This is placed at the very bottom so the full UI renders first,
#  then we wait, then trigger the next cycle.
# ================================================================
if st.session_state.running:
    time.sleep(UPDATE_INTERVAL_SEC)
    st.rerun()