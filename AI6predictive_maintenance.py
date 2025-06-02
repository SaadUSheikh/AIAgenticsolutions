
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample dataset
data = {
    "temperature": [70, 80, 75, 90, 85, 95],
    "vibration": [0.3, 0.4, 0.35, 0.5, 0.45, 0.55],
    "pressure": [10.5, 11.0, 9.8, 12.5, 11.5, 13.0],
    "failure": [0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Split data
X = df.drop("failure", axis=1)
y = df["failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Agent Implementation
class PredictiveMaintenanceAgent:
    def __init__(self, model):
        self.model = model

    def perceive(self, data_point):
        return np.array(data_point).reshape(1, -1)

    def decide(self, X):
        prediction = self.model.predict(X)[0]
        return "failure" if prediction == 1 else "no_failure"

    def act(self, decision):
        print(f"Maintenance needed: {decision}")

# Run agent
new_data_point = [85, 0.5, 12.8]
agent = PredictiveMaintenanceAgent(model=rf_model)
X = agent.perceive(new_data_point)
decision = agent.decide(X)
agent.act(decision)
