import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("data/tabular/Algerian_forest_fires_dataset.csv")

X = df.drop("fire", axis=1)
y = df["fire"]

# load trained model
model = joblib.load("models/xgboost_fire_model.pkl")

# predictions
y_pred = model.predict(X)

# accuracy
accuracy = accuracy_score(y, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y, y_pred))

# confusion matrix
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("XGBoost Weather Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("xgboost_confusion_matrix.png")

print("Confusion matrix saved.")