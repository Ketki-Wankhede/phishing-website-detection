
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Phishing_Legitimate_full.csv")
df = df.drop_duplicates()

# Prepare features and target
X = df.drop(columns=["CLASS_LABEL", "id"])
y = df["CLASS_LABEL"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Prepare values required by the app
medians = X_train.median().to_dict()

ranges = {
    col: (float(X_train[col].min()), float(X_train[col].max()))
    for col in X_train.columns
}

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
)
top_features = importance.nlargest(8).index.tolist()

# Save complete model bundle
bundle = {
    "model": model,
    "feature_names": list(X_train.columns),
    "all_columns": list(X_train.columns),
    "top_features": top_features,
    "medians": medians,
    "ranges": ranges,
}

joblib.dump(bundle, "random_forest_model.pkl")

print("Model bundle saved successfully.")
print("Adjustable features:", top_features)
