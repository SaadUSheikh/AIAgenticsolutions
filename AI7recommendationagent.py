
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Sample user-item interaction data
user_item_data = {
    "item1": [5, 4, 3, 0],
    "item2": [3, 0, 4, 5],
    "item3": [4, 3, 0, 5],
    "item4": [0, 5, 4, 3]
}

df_user_item = pd.DataFrame(user_item_data)

# Train KNN model
knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(df_user_item.values)

# Agent Implementation
class RecommendationAgent:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def perceive(self, user_id):
        return self.data.iloc[user_id].values.reshape(1, -1)

    def decide(self, user_vector):
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=2)
        recommendations = [self.data.index[i] for i in indices.flatten() if i != user_id]
        return recommendations

    def act(self, recommendations):
        print(f"Recommended items: {recommendations}")

# Run agent
user_id = 0
agent = RecommendationAgent(model=knn_model, data=df_user_item)
user_vector = agent.perceive(user_id)
recommendations = agent.decide(user_vector)
agent.act(recommendations)
