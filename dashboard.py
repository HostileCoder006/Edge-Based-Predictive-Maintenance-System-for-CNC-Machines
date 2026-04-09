"""
================================================================
  CNC Machine – Edge-Based Early Failure Warning System
  Streamlit Dashboard  ·  dashboard.py
================================================================
  Run with:      streamlit run dashboard.py
  Requires:      cnc_health_monitor.py + ml_model.py in same folder
  Install deps:  pip install streamlit pandas streamlit-autorefresh scikit-learn
================================================================
"""

import time
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ml_model import load_model
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

HISTORY_SIZE = 30

# ================================================================
#  PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="CNC Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
#  GLOBAL CSS
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    background-color: #080c12 !important;
    color: #cdd6e0;
    font-family: 'Inter', sans-serif;
    font-size: 15px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f18 0%, #080c12 100%) !important;
    border-right: 1px solid #1a2235;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

/* Keep Streamlit/Material icon ligatures from being shown as text (e.g. keyboard_double_*) */
span.material-symbols-rounded,
span.material-symbols-outlined,
span.material-symbols-sharp,
span.material-icons,
span.material-icons-round,
span.material-icons-outlined,
span.material-icons-sharp,
i.material-icons,
i.material-icons-round,
i.material-icons-outlined,
i.material-icons-sharp,
[class*="material-symbols"],
[class*="material-icons"],
[data-baseweb="icon"] span {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
}

/* Hide sidebar control ligature text like "keyboard_double_*" while preserving button behavior */
[data-testid="stSidebarCollapseButton"] span[class*="material"],
[data-testid="stSidebarCollapsedControl"] span[class*="material"],
[data-testid="stSidebarResizer"] span[class*="material"] {
    display: none !important;
}

/* Final fallback: hide Streamlit sidebar collapse controls entirely */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
}

[data-testid="stTabs"] { border-bottom: 1px solid #1a2235; }
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: #607080;
    padding: 0.5rem 1rem;
    transition: color 0.2s;
}
[data-testid="stTabs"] button:hover  { color: #a0b4c8; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4d9fff;
    font-weight: 600;
    border-bottom: 2px solid #4d9fff !important;
}

.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    border-radius: 8px;
    border: 1px solid #1e2d40;
    background: linear-gradient(135deg, #111827 0%, #0f1720 100%);
    color: #cdd6e0;
    padding: 0.55rem 1.5rem;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #162032 0%, #0f2030 100%);
    border-color: #4d9fff;
    color: #7dc4ff;
    box-shadow: 0 0 16px rgba(77,159,255,0.2);
    transform: translateY(-1px);
}

hr { border-color: #1a2235 !important; margin: 1.2rem 0 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080c12; }
::-webkit-scrollbar-thumb { background: #1a2235; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a3a55; }
</style>
""", unsafe_allow_html=True)

# ================================================================
#  SESSION STATE
# ================================================================
if "running"           not in st.session_state: st.session_state.running           = False
if "cycle"             not in st.session_state: st.session_state.cycle             = 0
if "history"           not in st.session_state: st.session_state.history           = pd.DataFrame(columns=["Cycle","Vibration","Temperature","Pressure","Sound","Health"])
if "latest"            not in st.session_state: st.session_state.latest            = None
if "shutdown"          not in st.session_state: st.session_state.shutdown          = False

if "ml_model" not in st.session_state:
    with st.spinner("⚙ Training ML model on startup…"):
        st.session_state.ml_model = load_model()

# ================================================================
#  STYLE HELPERS
# ================================================================
def ml_insight_top_sensor_live(model, data: dict, rule_health: float) -> str | None:
    """
    Counterfactual RF insight: which sensor most pulls ML health down vs nominal.
    Returns None when rule + ML agree things are healthy or the effect is too small
    (avoids bogus “degradation” at health 100).
    """
    names = ["Vibration", "Temperature", "Pressure", "Sound"]
    keys = ["vibration", "temperature", "pressure", "sound"]
    healthy_ref = {
        "vibration": 2.5,
        "temperature": 50.0,
        "pressure": 17.5,
        "sound": 60.0,
    }
    MIN_GAIN = 2.0  # points of predicted health; must clear this to name a stressor

    def as_row(d):
        return [[d["vibration"], d["temperature"], d["pressure"], d["sound"]]]

    ml_pred = max(0.0, min(100.0, float(model.predict(as_row(data))[0])))
    gains = []
    for k in keys:
        alt = dict(data)
        alt[k] = healthy_ref[k]
        pred = max(0.0, min(100.0, float(model.predict(as_row(alt))[0])))
        gains.append(pred - ml_pred)

    max_gain = max(gains)
    # Both scores in “healthy” band → no degradation headline
    if rule_health >= 90 and ml_pred >= 90:
        return None
    if max_gain < MIN_GAIN:
        return None
    return names[gains.index(max_gain)]


def section_label(text):
    st.markdown(f"""
    <div style='font-family:Inter,sans-serif;font-size:0.68rem;font-weight:600;
                color:#4a6080;letter-spacing:0.18em;text-transform:uppercase;
                margin-bottom:0.65rem;padding-bottom:0.3rem;
                border-bottom:1px solid #1a2235;'>{text}</div>
    """, unsafe_allow_html=True)


def failure_prediction_html(history_df: pd.DataFrame) -> str:
    """Trend from last 5–7 Health scores: high risk if declining or recent avg < 60."""
    if history_df.empty or len(history_df) < 5:
        return """
        <div style='background:#0d1422;border:1px solid #1a2235;border-radius:8px;
                    padding:0.55rem 1rem;margin:0.75rem 0 1rem;
                    font-family:Inter,sans-serif;font-size:0.82rem;color:#4a6080;'>
            📊 Collecting readings — failure prediction needs at least 5 cycles.
        </div>
        """
    h = history_df["Health"].astype(float).tail(7)
    avg = float(h.mean())
    diffs = h.diff().dropna()
    avg_diff = float(diffs.mean()) if len(diffs) else 0.0
    declining = avg_diff < 0
    high_risk = avg < 60 or declining
    if high_risk:
        return """
        <div style='background:linear-gradient(90deg,#1a1400,#120e00);
                    border:1px solid #eab308;border-left:4px solid #eab308;border-radius:8px;
                    padding:0.65rem 1.1rem;margin:0.75rem 0 1rem;
                    font-family:Inter,sans-serif;font-size:0.88rem;font-weight:600;
                    color:#eab308;'>
            ⚠️ High risk of failure soon
        </div>
        """
    return """
    <div style='background:linear-gradient(90deg,#071a10,#050f0a);
                border:1px solid #1a3a25;border-left:4px solid #22c55e;border-radius:8px;
                padding:0.65rem 1.1rem;margin:0.75rem 0 1rem;
                font-family:Inter,sans-serif;font-size:0.88rem;font-weight:600;
                color:#4db870;'>
        ✅ System stable
    </div>
    """


# ================================================================
#  SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.2rem;border-bottom:1px solid #1a2235;margin-bottom:1rem;'>
        <div style='font-family:Orbitron,monospace;font-size:0.7rem;font-weight:700;
                    color:#4d9fff;letter-spacing:0.22em;margin-bottom:0.3rem;'>⚙ CNC MONITOR</div>
        <div style='font-family:Inter,sans-serif;font-size:0.75rem;color:#4a6080;
                    letter-spacing:0.05em;'>Edge Failure Warning System</div>
    </div>
    """, unsafe_allow_html=True)

    def threshold_card(label, value, unit, color, icon):
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d1422 0%,#0b1018 100%);
                    border:1px solid #1a2235;border-left:3px solid {color};
                    border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.5rem;'>
            <div style='font-family:Inter,sans-serif;font-size:0.65rem;font-weight:500;
                        color:#4a6080;letter-spacing:0.1em;text-transform:uppercase;
                        margin-bottom:0.25rem;'>{icon} &nbsp;{label}</div>
            <div style='font-family:Inter,sans-serif;font-size:1rem;font-weight:700;color:#cdd6e0;'>
                {value}<span style='font-size:0.72rem;font-weight:400;color:#4a6080;margin-left:3px;'>{unit}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    section_label("Sensor Thresholds")
    threshold_card("Vibration",   VIBRATION_THRESHOLD,   "mm/s", "#3b82f6", "〰")
    threshold_card("Temperature", TEMPERATURE_THRESHOLD, "°C",   "#ef4444", "🌡")
    threshold_card("Pressure",    PRESSURE_THRESHOLD,    "bar",  "#f97316", "⬆")
    threshold_card("Sound",       SOUND_THRESHOLD,       "dB",   "#a855f7", "◉")

    st.markdown("<hr>", unsafe_allow_html=True)
    section_label("Score Legend")

    for color, glow, rng, label in [
        ("#22c55e", "rgba(34,197,94,0.15)",  "90 – 100", "HEALTHY"),
        ("#eab308", "rgba(234,179,8,0.15)",  "50 – 89",  "WARNING"),
        ("#ef4444", "rgba(239,68,68,0.15)",  "10 – 49",  "CRITICAL"),
        ("#dc2626", "rgba(220,38,38,0.1)",   "< 10",     "AUTO SHUTDOWN"),
    ]:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.6rem;
                    margin-bottom:0.4rem;padding:0.3rem 0.5rem;
                    border-radius:5px;background:{glow};'>
            <div style='width:8px;height:8px;border-radius:50%;
                        background:{color};box-shadow:0 0 6px {color};flex-shrink:0;'></div>
            <span style='font-family:Inter,sans-serif;font-size:0.78rem;color:#8a9ab0;'>
                {rng} &nbsp;<span style='color:{color};font-weight:600;'>{label}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:Inter,sans-serif;font-size:0.75rem;color:#4a6080;line-height:2.2;'>
        <span style='color:#607080;'>Update Interval</span> &nbsp;
        <span style='color:#8a9ab0;font-weight:600;'>{UPDATE_INTERVAL_SEC}s</span><br>
        <span style='color:#607080;'>History Window</span> &nbsp;
        <span style='color:#8a9ab0;font-weight:600;'>{HISTORY_SIZE} pts</span><br>
        <span style='color:#607080;'>Version</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <span style='color:#8a9ab0;font-weight:600;'>v2.3</span>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  HEADER
# ================================================================
st.markdown("""
<div style='background:linear-gradient(135deg,#0d1422 0%,#0a1020 50%,#0d1828 100%);
            border:1px solid #1a2235;border-left:4px solid #4d9fff;
            border-radius:10px;padding:1.3rem 1.8rem;margin-bottom:1.5rem;
            box-shadow:0 4px 24px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.03);'>
    <div style='display:flex;align-items:center;justify-content:space-between;'>
        <div>
            <div style='font-family:Inter,sans-serif;font-size:0.68rem;font-weight:600;
                        color:#4d9fff;letter-spacing:0.22em;text-transform:uppercase;
                        margin-bottom:0.35rem;'>
                Edge-Based Predictive Maintenance System
            </div>
            <div style='font-family:Orbitron,monospace;font-size:1.65rem;font-weight:700;
                        color:#e8f0f8;letter-spacing:0.06em;line-height:1.2;'>
                CNC Health Monitor
            </div>
            <div style='font-family:Inter,sans-serif;font-size:0.78rem;color:#4a6080;
                        margin-top:0.35rem;font-weight:400;'>
                Real-time anomaly detection &nbsp;·&nbsp; 4-sensor array &nbsp;·&nbsp; Auto-shutdown protection
            </div>
        </div>
        <div style='text-align:right;opacity:0.15;font-size:3rem;line-height:1;'>⚙</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
#  CONTROL ROW
# ================================================================
col_btn, col_status = st.columns([2, 5])

with col_btn:
    if st.session_state.shutdown:
        if st.button("🔄  Reset & Restart"):
            st.session_state.shutdown = False
            st.session_state.running  = True
    else:
        label = "⏹  Stop Monitoring" if st.session_state.running else "▶  Start Monitoring"
        if st.button(label):
            st.session_state.running = not st.session_state.running

with col_status:
    if st.session_state.running:
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#0b2218,#091a12);
                    border:1px solid #22c55e;border-radius:8px;
                    padding:0.58rem 1.1rem;display:inline-flex;align-items:center;gap:0.7rem;
                    box-shadow:0 0 20px rgba(34,197,94,0.12);'>
            <div style='width:8px;height:8px;border-radius:50%;background:#22c55e;
                        box-shadow:0 0 8px #22c55e;animation:pulse 1.5s infinite;'></div>
            <span style='font-family:Inter,sans-serif;font-size:0.82rem;font-weight:600;
                         color:#22c55e;letter-spacing:0.06em;'>LIVE</span>
            <span style='color:#2a5040;font-size:0.75rem;'>|</span>
            <span style='font-family:Orbitron,monospace;font-size:0.72rem;color:#4a9070;'>
                CYCLE&nbsp;#{st.session_state.cycle}</span>
            <span style='color:#2a5040;font-size:0.75rem;'>|</span>
            <span style='font-family:Orbitron,monospace;font-size:0.72rem;color:#4a9070;'>
                {time.strftime("%H:%M:%S")}</span>
        </div>
        <style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}</style>
        """, unsafe_allow_html=True)
    elif st.session_state.shutdown:
        st.markdown("""
        <div style='background:linear-gradient(90deg,#1a0808,#120505);
                    border:1px solid #ef4444;border-radius:8px;
                    padding:0.58rem 1.1rem;display:inline-flex;align-items:center;gap:0.7rem;
                    box-shadow:0 0 20px rgba(239,68,68,0.15);'>
            <span style='font-size:1rem;'>⛔</span>
            <span style='font-family:Inter,sans-serif;font-size:0.82rem;font-weight:600;
                         color:#ef4444;letter-spacing:0.06em;'>SHUTDOWN ACTIVE</span>
            <span style='color:#502020;'>|</span>
            <span style='font-family:Inter,sans-serif;font-size:0.75rem;color:#804040;'>
                Awaiting reset &amp; inspection</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0d1220;border:1px solid #1a2235;border-radius:8px;
                    padding:0.58rem 1.1rem;display:inline-flex;align-items:center;gap:0.7rem;'>
            <div style='width:8px;height:8px;border-radius:50%;background:#2a3a55;'></div>
            <span style='font-family:Inter,sans-serif;font-size:0.82rem;font-weight:500;
                         color:#4a6080;letter-spacing:0.05em;'>STOPPED — Press Start to begin monitoring</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ================================================================
#  DATA COLLECTION
# ================================================================
if st.session_state.running:
    st.session_state.cycle += 1

    data          = generate_sensor_data()
    anomalies     = detect_anomalies(data)
    score, status = calculate_health_score(data, anomalies)
    suggestions   = get_suggestions(anomalies)

    st.session_state.latest = {
        "data": data, "anomalies": anomalies,
        "score": score, "status": status,
        "suggestions": suggestions,
    }

    if score < 10:
        st.session_state.running  = False
        st.session_state.shutdown = True

    new_row = pd.DataFrame([{
        "Cycle":       st.session_state.cycle,
        "Vibration":   data["vibration"],
        "Temperature": data["temperature"],
        "Pressure":    data["pressure"],
        "Sound":       data["sound"],
        "Health":      score,
    }])
    if st.session_state.history.empty:
        st.session_state.history = new_row
    else:
        st.session_state.history = pd.concat(
            [st.session_state.history, new_row], ignore_index=True
        ).tail(HISTORY_SIZE)

# ================================================================
#  UI RENDER
# ================================================================
snap = st.session_state.latest

if snap is not None:
    data        = snap["data"]
    anomalies   = snap["anomalies"]
    score       = snap["score"]
    status      = snap["status"]
    suggestions = snap["suggestions"]

    # ── STATUS STYLES ───────────────────────────────────────
    STATUS_STYLES = {
        "Healthy":  {"color": "#22c55e", "glow": "rgba(34,197,94,0.18)",  "bg": "#071a10", "icon": "✦"},
        "Warning":  {"color": "#eab308", "glow": "rgba(234,179,8,0.18)",  "bg": "#1a1400", "icon": "▲"},
        "Critical": {"color": "#ef4444", "glow": "rgba(239,68,68,0.18)",  "bg": "#1a0505", "icon": "✖"},
    }
    S = STATUS_STYLES.get(status, STATUS_STYLES["Critical"])

    # ── HEALTH SCORE BANNER ─────────────────────────────────
    bar_pct   = max(0, min(100, score))
    bar_color = S["color"]

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{S["bg"]} 0%,#0b0f18 100%);
                border:1px solid {S["color"]}40;border-left:5px solid {S["color"]};
                border-radius:10px;padding:1.2rem 1.8rem;margin-bottom:1.3rem;
                box-shadow:0 0 30px {S["glow"]},0 4px 16px rgba(0,0,0,0.5);'>
        <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:0.9rem;'>
            <div>
                <div style='font-family:Inter,sans-serif;font-size:0.68rem;font-weight:600;
                            color:{S["color"]}99;letter-spacing:0.18em;text-transform:uppercase;
                            margin-bottom:0.3rem;'>Machine Status</div>
                <div style='font-family:Orbitron,monospace;font-size:1.8rem;font-weight:700;
                            color:{S["color"]};letter-spacing:0.05em;
                            text-shadow:0 0 20px {S["color"]}60;'>
                    {S["icon"]} &nbsp;{status.upper()}
                </div>
            </div>
            <div style='text-align:right;'>
                <div style='font-family:Inter,sans-serif;font-size:0.65rem;font-weight:500;
                            color:#4a6080;letter-spacing:0.12em;text-transform:uppercase;
                            margin-bottom:0.15rem;'>Health Score</div>
                <div style='font-family:Orbitron,monospace;font-size:3.4rem;font-weight:900;
                            color:{S["color"]};line-height:1;
                            text-shadow:0 0 30px {S["color"]}80;'>{score}</div>
                <div style='font-family:Inter,sans-serif;font-size:0.7rem;
                            color:#4a6080;margin-top:0.1rem;'>out of 100</div>
            </div>
        </div>
        <div style='background:#0d1422;border-radius:4px;height:6px;overflow:hidden;border:1px solid #1a2235;'>
            <div style='height:100%;width:{bar_pct}%;
                        background:linear-gradient(90deg,{bar_color}80,{bar_color});
                        border-radius:4px;transition:width 0.5s ease;
                        box-shadow:0 0 8px {bar_color}80;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;
                    font-family:Inter,sans-serif;font-size:0.6rem;color:#2a3a55;margin-top:0.25rem;'>
            <span>0 — SHUTDOWN</span><span>50 — WARNING</span><span>90 — HEALTHY — 100</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(failure_prediction_html(st.session_state.history), unsafe_allow_html=True)

else:
    # ── IDLE STATE ───────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1422,#090d16);
                border:1px dashed #1a2235;border-radius:10px;
                padding:3.5rem;text-align:center;margin:0.5rem 0 1.5rem;'>
        <div style='font-size:2.5rem;margin-bottom:0.8rem;opacity:0.2;'>⚙</div>
        <div style='font-family:Orbitron,monospace;font-size:0.8rem;font-weight:600;
                    color:#2a3a55;letter-spacing:0.18em;margin-bottom:0.4rem;'>
            AWAITING SENSOR DATA
        </div>
        <div style='font-family:Inter,sans-serif;font-size:0.78rem;color:#2a3a55;'>
            Press <strong style="color:#4a6080;">▶ Start Monitoring</strong> to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  CHARTS
# ================================================================
st.markdown("<hr>", unsafe_allow_html=True)
section_label("System Trends")

df = st.session_state.history

if not df.empty:
    chart_df = df.set_index("Cycle")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Health Score", "Vibration", "Temperature", "Pressure", "Sound"
    ])
    with tab1: st.line_chart(chart_df[["Health"]],      height=180, width="stretch")
    with tab2: st.line_chart(chart_df[["Vibration"]],   height=180, width="stretch")
    with tab3: st.line_chart(chart_df[["Temperature"]], height=180, width="stretch")
    with tab4: st.line_chart(chart_df[["Pressure"]],    height=180, width="stretch")
    with tab5: st.line_chart(chart_df[["Sound"]],       height=180, width="stretch")
else:
    st.markdown("""
    <div style='background:#0d1422;border:1px dashed #1a2235;border-radius:8px;
                padding:1.2rem;text-align:center;font-family:Inter,sans-serif;
                font-size:0.78rem;color:#2a3a55;letter-spacing:0.06em;'>
        Trend data will appear after the first reading
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  LIVE SENSOR READINGS
# ================================================================
if snap is not None:
    section_label("Live Sensor Readings")

    sensors = [
        ("Vibration",   data["vibration"],   anomalies["vibration"],   "mm/s", "#3b82f6", "V", VIBRATION_THRESHOLD),
        ("Temperature", data["temperature"], anomalies["temperature"], "°C",   "#ef4444", "T", TEMPERATURE_THRESHOLD),
        ("Pressure",    data["pressure"],    anomalies["pressure"],    "bar",  "#f97316", "P", PRESSURE_THRESHOLD),
        ("Sound",       data["sound"],       anomalies["sound"],       "dB",   "#a855f7", "S", SOUND_THRESHOLD),
    ]

    cols = st.columns(4)
    for col, (name, val, is_anomaly, unit, accent, icon, thresh) in zip(cols, sensors):
        border_col = "#ef4444" if is_anomaly else accent
        glow_col   = "rgba(239,68,68,0.18)" if is_anomaly else "rgba(0,0,0,0.1)"
        tag_color  = "#ef4444" if is_anomaly else "#22c55e"
        tag_bg     = "#1a0505" if is_anomaly else "#071a10"
        tag_text   = "⚠ OVER LIMIT" if is_anomaly else "✔ NORMAL"
        val_color  = "#ef4444" if is_anomaly else "#e8f0f8"

        col.markdown(f"""
        <div style='background:linear-gradient(160deg,#0d1422 0%,#090d16 100%);
                    border:1px solid #1a2235;border-top:3px solid {border_col};
                    border-radius:10px;padding:1.1rem 1.2rem 1rem;
                    box-shadow:0 0 20px {glow_col},0 4px 12px rgba(0,0,0,0.4);'>
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:0.55rem;'>
                <div style='font-family:Inter,sans-serif;font-size:0.68rem;font-weight:600;
                            color:#4a6080;letter-spacing:0.12em;text-transform:uppercase;'>{icon} {name}</div>
                <div style='font-family:Inter,sans-serif;font-size:0.6rem;font-weight:600;
                            color:{tag_color};background:{tag_bg};
                            padding:0.15rem 0.5rem;border-radius:20px;
                            border:1px solid {tag_color}50;letter-spacing:0.06em;'>{tag_text}</div>
            </div>
            <div style='font-family:Orbitron,monospace;font-size:2rem;font-weight:700;
                        color:{val_color};line-height:1;letter-spacing:0.03em;
                        {"text-shadow:0 0 16px #ef444480;" if is_anomaly else ""}'>
                {val}
            </div>
            <div style='font-family:Inter,sans-serif;font-size:0.72rem;color:#4a6080;margin-top:0.25rem;'>
                {unit}<span style='color:#2a3a55;margin-left:0.4rem;'>· limit {thresh}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
#  MAINTENANCE SUGGESTIONS
# ================================================================
if snap is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Maintenance Suggestions")

    if suggestions:
        s_cols = st.columns(len(suggestions))
        for col, s in zip(s_cols, suggestions):
            col.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a1400,#110e00);
                        border:1px solid #3a2a00;border-left:4px solid #eab308;
                        border-radius:8px;padding:0.85rem 1rem;
                        box-shadow:0 0 16px rgba(234,179,8,0.08);'>
                <div style='font-size:1.1rem;margin-bottom:0.35rem;'>🔧</div>
                <div style='font-family:Inter,sans-serif;font-size:0.85rem;font-weight:500;
                            color:#d4aa30;line-height:1.5;'>{s}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#071a10,#050f0a);
                    border:1px solid #1a3a25;border-left:4px solid #22c55e;
                    border-radius:8px;padding:0.85rem 1.2rem;
                    box-shadow:0 0 16px rgba(34,197,94,0.08);
                    display:flex;align-items:center;gap:0.7rem;'>
            <span style='font-size:1.1rem;'>✅</span>
            <span style='font-family:Inter,sans-serif;font-size:0.88rem;font-weight:500;
                         color:#4db870;'>All systems nominal — no maintenance action required</span>
        </div>
        """, unsafe_allow_html=True)

    top_sensor = ml_insight_top_sensor_live(st.session_state.ml_model, data, score)
    if top_sensor is None:
        st.markdown("""
        <div style='background:#0b1420;border:1px solid #1a3a35;border-left:4px solid #22c55e;
                    border-radius:8px;padding:0.75rem 1.1rem;margin-top:0.75rem;
                    font-family:Inter,sans-serif;font-size:0.88rem;color:#6eb889;line-height:1.5;'>
            📊 AI Based Insights: <strong style="color:#4db870;">No dominant degradation signal</strong> — readings match healthy operation
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#0b1420;border:1px solid #1e3a5a;border-left:4px solid #4d9fff;
                    border-radius:8px;padding:0.75rem 1.1rem;margin-top:0.75rem;
                    font-family:Inter,sans-serif;font-size:0.88rem;color:#8ab4e8;line-height:1.5;'>
            📊 AI Based Insights: <strong style="color:#7dc4ff;">{top_sensor}</strong> contributing most to degradation
        </div>
        """, unsafe_allow_html=True)

# ================================================================
#  AUTO-SHUTDOWN BANNER
# ================================================================
if st.session_state.shutdown:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a0505,#0f0303);
                border:2px solid #ef4444;border-radius:10px;
                padding:1.5rem 2rem;margin-top:1.4rem;
                box-shadow:0 0 40px rgba(239,68,68,0.25),0 4px 20px rgba(0,0,0,0.5);'>
        <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;'>
            <span style='font-size:1.5rem;'>🚨</span>
            <div style='font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;
                        color:#ef4444;letter-spacing:0.1em;
                        text-shadow:0 0 20px rgba(239,68,68,0.6);'>
                SYSTEM SHUTDOWN TRIGGERED
            </div>
        </div>
        <div style='font-family:Inter,sans-serif;font-size:0.9rem;color:#c07070;
                    font-weight:400;line-height:1.7;padding-left:2.3rem;'>
            Health score fell below <strong style='color:#ef4444;'>10 / 100</strong> — 
            machine halted automatically to prevent damage.<br>
            <span style='color:#ef4444;font-weight:600;'>
                Do not restart until a full physical inspection is completed.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  AUTO-REFRESH
#  Frontend timer-based refresh prevents white-screen blocking.
# ================================================================
elif st.session_state.running:
    st.markdown(
        f"<div style='font-family:Inter,sans-serif;font-size:0.7rem;"
        f"color:#2a3a55;text-align:right;padding-right:0.5rem;'>"
        f"Refreshing every {UPDATE_INTERVAL_SEC}s...</div>",
        unsafe_allow_html=True,
    )
    st_autorefresh(interval=UPDATE_INTERVAL_SEC * 1000, key="monitor_refresh")
