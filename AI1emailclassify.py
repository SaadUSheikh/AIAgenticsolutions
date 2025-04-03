from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample email dataset different emails and labels
emails = [
    "Hi Saad how are you , i want to do a startup", 
    "Hi Bob, how are you?", 
    "Limited time offer!", 
    "Hello friend, just checking in.",
    "Looking for sales come partner with us"

]
labels = [1, 0, 1, 0 ,1]  # 1: Spam, 0: Not Spam

# Convert text to numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
