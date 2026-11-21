
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("fruit_quality.csv")
X = data.drop("Quality", axis=1)
y = data["Quality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Decision Tree Model Training Completed")
print("Accuracy:", accuracy * 100, "%")
print(classification_report(y_test, predictions))
with open("fruit_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("Model saved successfully!")
