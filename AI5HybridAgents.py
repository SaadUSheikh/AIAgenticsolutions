import numpy as np
from sklearn.linear_model import LinearRegression

# === Sample sales data (price, ad spend, season) and actual sales ===
sales_data = np.array([
    [10, 1000, 1], [12, 1500, 2],
    [11, 1200, 3], [13, 1600, 4]
])
sales = np.array([500, 600, 550, 700])

# === Train linear regression model ===
demand_model = LinearRegression()
demand_model.fit(sales_data, sales)

# === Dynamic Pricing Agent ===
class DynamicPricingAgent:
    def __init__(self, demand_model):
        self.demand_model = demand_model

    def perceive(self, features):
        return np.array(features).reshape(1, -1)

    def decide(self, X):
        forecasted_demand = self.demand_model.predict(X)[0]
        if forecasted_demand > 600:
            return "increase_price"
        elif forecasted_demand < 550:
            return "decrease_price"
        else:
            return "maintain_price"

    def act(self, decision):
        print(f"Pricing decision: {decision}")

# === Sample usage ===
new_features = [11, 1300, 1]
agent = DynamicPricingAgent(demand_model=demand_model)

X = agent.perceive(new_features)
decision = agent.decide(X)
agent.act(decision)
