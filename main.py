import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.models import (
    train_logistic_regression,
    train_decision_tree
)

from src.evaluation import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    create_metrics_dataframe,
    plot_roc_curves
)

from src.cross_validation import (
    create_cross_validation_dataframe
)


# ==========================================
# 1. Create Output Directories
# ==========================================

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# ==========================================
# 2. Load Week 3 Input Data
# ==========================================

print("=" * 60)
print("PREDICTIVE PERFORMANCE AUDIT CONSOLE")
print("=" * 60)

print("\n========== DATA LOADING ==========")

X = pd.read_csv("data/final_feature_matrix.csv")
y = pd.read_csv("data/target_churn.csv").squeeze()

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 3. Verify Input Data
# ==========================================

print("\n========== DATA VERIFICATION ==========")

print(f"Number of records: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Missing values: {X.isnull().sum().sum()}")
print(f"Duplicate rows: {X.duplicated().sum()}")


# ==========================================
# 4. Train-Test Split
# ==========================================

print("\n========== TRAIN-TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training data: {X_train.shape}")
print(f"Testing data:  {X_test.shape}")


# ==========================================
# 5. Train Logistic Regression
# ==========================================

print("\n========== MODEL TRAINING ==========")

logistic_model = train_logistic_regression(
    X_train,
    y_train
)

print("Logistic Regression trained successfully.")


# ==========================================
# 6. Train Decision Tree
# ==========================================

decision_tree_model = train_decision_tree(
    X_train,
    y_train
)

print("Decision Tree trained successfully.")


# ==========================================
# 7. Calculate Logistic Regression Metrics
# ==========================================

print("\n========== LOGISTIC REGRESSION AUDIT ==========")

logistic_metrics, logistic_predictions, logistic_probabilities = (
    calculate_classification_metrics(
        logistic_model,
        X_test,
        y_test
    )
)

logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_model.predict(X_train)
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print(f"Training Accuracy: {logistic_train_accuracy:.4f}")
print(f"Testing Accuracy:  {logistic_test_accuracy:.4f}")

for metric, value in logistic_metrics.items():
    print(f"{metric}: {value:.4f}")


# ==========================================
# 8. Calculate Decision Tree Metrics
# ==========================================

print("\n========== DECISION TREE AUDIT ==========")

tree_metrics, tree_predictions, tree_probabilities = (
    calculate_classification_metrics(
        decision_tree_model,
        X_test,
        y_test
    )
)

tree_train_accuracy = accuracy_score(
    y_train,
    decision_tree_model.predict(X_train)
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

print(f"Training Accuracy: {tree_train_accuracy:.4f}")
print(f"Testing Accuracy:  {tree_test_accuracy:.4f}")

for metric, value in tree_metrics.items():
    print(f"{metric}: {value:.4f}")


# ==========================================
# 9. Create Model Metrics Report
# ==========================================

metrics_df = create_metrics_dataframe(
    logistic_metrics,
    tree_metrics
)

metrics_df["Training Accuracy"] = [
    logistic_train_accuracy,
    tree_train_accuracy
]

metrics_df["Testing Accuracy"] = [
    logistic_test_accuracy,
    tree_test_accuracy
]

metrics_df = metrics_df[
    [
        "Model",
        "Training Accuracy",
        "Testing Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
]

print("\n========== MODEL METRIC COMPARISON ==========")
print(metrics_df.to_string(index=False))

metrics_df.to_csv(
    "outputs/reports/model_metrics.csv",
    index=False
)

print("\nModel metrics report saved successfully.")


# ==========================================
# 10. Generate Confusion Matrices
# ==========================================

print("\n========== CONFUSION MATRICES ==========")

plot_confusion_matrix(
    y_test,
    logistic_predictions,
    "Logistic Regression",
    "outputs/figures/logistic_regression_confusion_matrix.png"
)

plot_confusion_matrix(
    y_test,
    tree_predictions,
    "Decision Tree",
    "outputs/figures/decision_tree_confusion_matrix.png"
)

print("Logistic Regression confusion matrix saved.")
print("Decision Tree confusion matrix saved.")


# ==========================================
# 10.5 Generate ROC Curve Comparison
# ==========================================

print("\n========== ROC CURVE ==========")

plot_roc_curves(
    y_test,
    logistic_probabilities,
    tree_probabilities,
    "outputs/figures/roc_curve_comparison.png"
)

print("ROC curve comparison saved.")


# ==========================================
# 11. Run 5-Fold Cross-Validation
# ==========================================

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

cv_results = create_cross_validation_dataframe(
    logistic_model,
    decision_tree_model,
    X,
    y
)

print(cv_results.to_string(index=False))

cv_results.to_csv(
    "outputs/reports/cross_validation_results.csv",
    index=False
)

print("\nCross-validation report saved successfully.")


# ==========================================
# 12. Overfitting Audit
# ==========================================

print("\n========== OVERFITTING AUDIT ==========")

audit_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Training Accuracy": [
        logistic_train_accuracy,
        tree_train_accuracy
    ],
    "Testing Accuracy": [
        logistic_test_accuracy,
        tree_test_accuracy
    ],
    "CV Mean F1": [
        cv_results.loc[
            cv_results["Model"] == "Logistic Regression",
            "Mean F1"
        ].iloc[0],
        cv_results.loc[
            cv_results["Model"] == "Decision Tree",
            "Mean F1"
        ].iloc[0]
    ],
    "CV Std F1": [
        cv_results.loc[
            cv_results["Model"] == "Logistic Regression",
            "Std F1"
        ].iloc[0],
        cv_results.loc[
            cv_results["Model"] == "Decision Tree",
            "Std F1"
        ].iloc[0]
    ]
})

audit_df["Train-Test Gap"] = (
    audit_df["Training Accuracy"]
    - audit_df["Testing Accuracy"]
)

print(audit_df.to_string(index=False))

audit_df.to_csv(
    "outputs/reports/performance_audit.csv",
    index=False
)

print("\nPerformance audit report saved successfully.")


# ==========================================
# 13. Final Summary
# ==========================================

print("\n" + "=" * 60)
print("WEEK 4 PERFORMANCE AUDIT COMPLETED")
print("=" * 60)

print("\nGenerated reports:")
print("- outputs/reports/model_metrics.csv")
print("- outputs/reports/cross_validation_results.csv")
print("- outputs/reports/performance_audit.csv")

print("\nGenerated figures:")
print("- outputs/figures/logistic_regression_confusion_matrix.png")
print("- outputs/figures/decision_tree_confusion_matrix.png")
print("- outputs/figures/roc_curve_comparison.png")

print("\nPipeline execution completed successfully.")