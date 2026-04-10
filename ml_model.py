import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor

VIBRATION_THRESHOLD   = 5.0
TEMPERATURE_THRESHOLD = 75.0
PRESSURE_THRESHOLD    = 30.0
SOUND_THRESHOLD       = 85.0
VIBRATION_PENALTY     = 3.0
TEMPERATURE_PENALTY   = 1.5
PRESSURE_PENALTY      = 2.0
SOUND_PENALTY         = 1.2


def _rule_score(vibration, temperature, pressure, sound):
    score = 100.0
    if vibration   > VIBRATION_THRESHOLD:
        score -= (vibration   - VIBRATION_THRESHOLD)   * VIBRATION_PENALTY
    if temperature > TEMPERATURE_THRESHOLD:
        score -= (temperature - TEMPERATURE_THRESHOLD) * TEMPERATURE_PENALTY
    if pressure    > PRESSURE_THRESHOLD:
        score -= (pressure    - PRESSURE_THRESHOLD)    * PRESSURE_PENALTY
    if sound       > SOUND_THRESHOLD:
        score -= (sound       - SOUND_THRESHOLD)       * SOUND_PENALTY
    return max(0.0, min(100.0, score))


def _generate_training_data(n_samples=2000):
    X, y = [], []
    for _ in range(n_samples):
        is_fault = random.random() < 0.25
        if is_fault:
            v = random.uniform(5.0,  12.0)
            t = random.uniform(75.0, 100.0)
            p = random.uniform(30.0,  50.0)
            s = random.uniform(85.0, 110.0)
        else:
            v = random.uniform(0.5,   4.5)
            t = random.uniform(30.0,  70.0)
            p = random.uniform(10.0,  25.0)
            s = random.uniform(40.0,  80.0)
        score = _rule_score(v, t, p, s)
        score = max(0.0, min(100.0, score + random.gauss(0, 2.5)))
        X.append([v, t, p, s])
        y.append(score)
    return np.array(X), np.array(y)


def load_model():
    X, y = _generate_training_data(n_samples=2000)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


def ml_predict(model, data: dict) -> float:
    features = np.array([[data["vibration"], data["temperature"], data["pressure"], data["sound"]]])
    score = model.predict(features)[0]
    return round(float(max(0.0, min(100.0, score))), 1)
