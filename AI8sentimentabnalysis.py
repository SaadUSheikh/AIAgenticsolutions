
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample text data and labels
texts = ["I love this product!", "This is the worst experience ever.", "Absolutely fantastic!", "Not worth the money."]
labels = [1, 0, 1, 0]

# Convert text data to numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X, y)

# Agent Implementation
class SentimentAnalysisAgent:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def perceive(self, text):
        return self.vectorizer.transform([text])

    def decide(self, X):
        prediction = self.model.predict(X)[0]
        return "positive" if prediction == 1 else "negative"

    def act(self, decision):
        print(f"Sentiment: {decision}")

# Run agent
new_text = "I had a great time using this service."
agent = SentimentAnalysisAgent(model=nb_model, vectorizer=vectorizer)
X = agent.perceive(new_text)
decision = agent.decide(X)
agent.act(decision)
