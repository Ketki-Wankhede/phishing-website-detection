
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 1. Load dataset
df = pd.read_csv("Phishing_Legitimate_full.csv")

# 2. Remove duplicate records, if any
df = df.drop_duplicates().copy()

# 3. Separate features and target
# ID is an identifier, not a useful prediction feature.
X = df.drop(columns=["CLASS_LABEL", "id"])
y = df["CLASS_LABEL"].astype(int)

print("Dataset shape after cleaning:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("\nTarget distribution:")
print(y.value_counts().sort_index())

# 4. Stratified 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 5. Define four models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
}

results = []
fitted_models = {}

# 6. Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(scores)
    fitted_models[name] = model

    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate (0)", "Phishing (1)"],
        zero_division=0,
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Legitimate", "Phishing"],
        cmap="Blues",
        values_format="d",
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_confusion.png"
    plt.savefig(filename, dpi=150)
    plt.close()

# 7. Save and display comparison table
results_df = pd.DataFrame(results).sort_values(
    by="F1 Score", ascending=False
)
results_df.to_csv("model_comparison.csv", index=False)

print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved results to model_comparison.csv")
print("Saved confusion matrix images for all four models.")
