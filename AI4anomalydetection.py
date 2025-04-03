import numpy as np
from sklearn.ensemble import IsolationForest

# === Sample network traffic data (duration, bytes transferred) ===
traffic_data = np.array([
    [100, 2000], [110, 2200], [95, 2100],
    [50, 8000], [120, 2400], [130, 2500]
])

# === Train Isolation Forest ===
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(traffic_data)

# === Anomaly Detection Agent ===
class AnomalyDetectionAgent:
    def __init__(self, model):
        self.model = model

    def perceive(self, data_point):
        # Reshape input
        return np.array(data_point).reshape(1, -1)

    def decide(self, X):
        prediction = self.model.predict(X)
        return "anomaly" if prediction == -1 else "normal"

    def act(self, decision):
        print(f"Network traffic classified as: {decision}")

# === Sample usage ===
new_data_point = [60, 9000]
agent = AnomalyDetectionAgent(model=iso_forest)

X = agent.perceive(new_data_point)
decision = agent.decide(X)
agent.act(decision)
