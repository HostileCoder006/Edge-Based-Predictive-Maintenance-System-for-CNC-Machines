"""
========================================================
  Edge-Based Early Failure Warning System
  Health Scoring for CNC Machines (Enhanced Version)
========================================================
  Features:
  - Vibration, Temperature, Pressure, Sound sensors
  - Improved health scoring
  - Maintenance suggestions
========================================================
"""

import random
import time
import sys

# ── CONFIGURATION ──────────────────────────

VIBRATION_THRESHOLD   = 5.0    # mm/s
TEMPERATURE_THRESHOLD = 75.0   # °C
PRESSURE_THRESHOLD    = 30.0   # bar
SOUND_THRESHOLD       = 85.0   # dB  ← NEW

UPDATE_INTERVAL_SEC   = 2

# Penalties per unit over threshold
VIBRATION_PENALTY   = 3.0
TEMPERATURE_PENALTY = 1.5
PRESSURE_PENALTY    = 2.0
SOUND_PENALTY       = 1.2      # ← NEW


# ═══════════════════════════════════════════
# 1. SENSOR DATA
# ═══════════════════════════════════════════

def generate_sensor_data():
    """
    Simulates all four CNC sensors.
    25% chance of a fault condition.
    """
    is_fault = random.random() < 0.25

    if is_fault:
        vibration   = round(random.uniform(5.0,  12.0), 2)
        temperature = round(random.uniform(75.0, 100.0), 2)
        pressure    = round(random.uniform(30.0,  50.0), 2)
        sound       = round(random.uniform(85.0, 110.0), 2)
    else:
        vibration   = round(random.uniform(0.5,   4.5), 2)
        temperature = round(random.uniform(30.0,  70.0), 2)
        pressure    = round(random.uniform(10.0,  25.0), 2)
        sound       = round(random.uniform(40.0,  80.0), 2)

    return {
        "vibration":   vibration,
        "temperature": temperature,
        "pressure":    pressure,
        "sound":       sound,
    }


# ═══════════════════════════════════════════
# 2. ANOMALY DETECTION
# ═══════════════════════════════════════════

def detect_anomalies(data):
    """Returns True for each sensor that exceeds its threshold."""
    return {
        "vibration":   data["vibration"]   > VIBRATION_THRESHOLD,
        "temperature": data["temperature"] > TEMPERATURE_THRESHOLD,
        "pressure":    data["pressure"]    > PRESSURE_THRESHOLD,
        "sound":       data["sound"]       > SOUND_THRESHOLD,
    }


# ═══════════════════════════════════════════
# 3. HEALTH SCORE
# ═══════════════════════════════════════════

def calculate_health_score(data, anomalies):
    """
    Starts at 100 and deducts penalty points proportional
    to how far each sensor exceeds its threshold.
    """
    score = 100.0

    if anomalies["vibration"]:
        score -= (data["vibration"]   - VIBRATION_THRESHOLD)   * VIBRATION_PENALTY

    if anomalies["temperature"]:
        score -= (data["temperature"] - TEMPERATURE_THRESHOLD) * TEMPERATURE_PENALTY

    if anomalies["pressure"]:
        score -= (data["pressure"]    - PRESSURE_THRESHOLD)    * PRESSURE_PENALTY

    if anomalies["sound"]:
        score -= (data["sound"]       - SOUND_THRESHOLD)       * SOUND_PENALTY

    score = max(0.0, min(100.0, score))

    if score >= 90:
        status = "Healthy"
    elif score >= 70:
        status = "Warning"
    else:
        status = "Critical"

    return round(score, 1), status


# ═══════════════════════════════════════════
# 4. MAINTENANCE SUGGESTIONS
# ═══════════════════════════════════════════

def get_suggestions(anomalies):
    """Returns a list of actionable maintenance tips."""
    suggestions = []

    if anomalies["vibration"]:
        suggestions.append("Check bearings or spindle alignment")

    if anomalies["temperature"]:
        suggestions.append("Inspect cooling system / coolant level")

    if anomalies["pressure"]:
        suggestions.append("Check hydraulic / pneumatic lines")

    if anomalies["sound"]:
        suggestions.append("Inspect for loose parts or tool wear (high noise)")

    return suggestions


# ═══════════════════════════════════════════
# 5. CONSOLE DISPLAY (text-only mode)
# ═══════════════════════════════════════════

def display(data, score, status, suggestions):
    print("\n" + "=" * 52)
    print("  CNC MACHINE HEALTH MONITOR")
    print("=" * 52)
    print(f"  Vibration   : {data['vibration']}  mm/s  (threshold: {VIBRATION_THRESHOLD})")
    print(f"  Temperature : {data['temperature']}  °C    (threshold: {TEMPERATURE_THRESHOLD})")
    print(f"  Pressure    : {data['pressure']}  bar   (threshold: {PRESSURE_THRESHOLD})")
    print(f"  Sound       : {data['sound']}  dB    (threshold: {SOUND_THRESHOLD})")
    print(f"\n  Health Score: {score}  →  {status}")

    if suggestions:
        print("\n  Maintenance Suggestions:")
        for s in suggestions:
            print(f"    - {s}")
    else:
        print("\n  All systems normal")

    print("=" * 52)


# ═══════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════

def run():
    try:
        while True:
            data            = generate_sensor_data()
            anomalies       = detect_anomalies(data)
            score, status   = calculate_health_score(data, anomalies)
            suggestions     = get_suggestions(anomalies)
            display(data, score, status, suggestions)
            time.sleep(UPDATE_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        sys.exit()


if __name__ == "__main__":
    run()