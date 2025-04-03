from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# === 1. Dataset ===
emails = [
    "Free money now!!!",
    "Hi Bob, how are you?",
    "Limited time offer!",
    "Hello friend, just checking in."
]
labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# === 2. Text vectorization ===
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = labels

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# === 4. Train classifier ===
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# === 5. Agent class ===
class SpamDetectionAgent:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def perceive(self, email):
        # Convert email text to numerical data
        X = self.vectorizer.transform([email])
        return X

    def decide(self, X):
        # Use trained model to predict spam or not
        prediction = self.model.predict(X)
        return "spam" if prediction == 1 else "not_spam"

    def act(self, decision):
        print(f"Email classified as: {decision}")

# === 6. Sample usage ===
new_email = "Win a free iPhone now!!!"
agent = SpamDetectionAgent(model=clf, vectorizer=vectorizer)

# Agent perception-decision-action cycle
X = agent.perceive(new_email)
decision = agent.decide(X)
agent.act(decision)
