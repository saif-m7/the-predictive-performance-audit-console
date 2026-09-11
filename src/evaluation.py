import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def calculate_classification_metrics(model, X_test, y_test):
    """
    Calculate classification performance metrics.
    """

    # Class predictions
    y_pred = model.predict(X_test)

    # Probability of positive class for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    return metrics, y_pred, y_prob


def plot_confusion_matrix(
    y_test,
    y_pred,
    model_name,
    output_path
):
    """
    Generate and save a confusion matrix plot.
    """

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churn", "Churn"],
        yticklabels=["Not Churn", "Churn"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


def create_metrics_dataframe(
    logistic_metrics,
    tree_metrics
):
    """
    Create a comparison DataFrame for both models.
    """

    metrics_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree"
        ],
        "Precision": [
            logistic_metrics["Precision"],
            tree_metrics["Precision"]
        ],
        "Recall": [
            logistic_metrics["Recall"],
            tree_metrics["Recall"]
        ],
        "F1-Score": [
            logistic_metrics["F1-Score"],
            tree_metrics["F1-Score"]
        ],
        "ROC-AUC": [
            logistic_metrics["ROC-AUC"],
            tree_metrics["ROC-AUC"]
        ]
    })

    return metrics_df


def plot_roc_curves(
    y_test,
    logistic_probabilities,
    tree_probabilities,
    output_path
):
    """
    Generate and save ROC curves for both models.
    """

    # Calculate ROC curve for Logistic Regression
    logistic_fpr, logistic_tpr, _ = roc_curve(
        y_test,
        logistic_probabilities
    )

    # Calculate ROC curve for Decision Tree
    tree_fpr, tree_tpr, _ = roc_curve(
        y_test,
        tree_probabilities
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        logistic_fpr,
        logistic_tpr,
        label="Logistic Regression"
    )

    plt.plot(
        tree_fpr,
        tree_tpr,
        label="Decision Tree"
    )

    # Random classifier reference line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()