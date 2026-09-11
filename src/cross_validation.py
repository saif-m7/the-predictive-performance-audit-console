import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score


def run_cross_validation(model, X, y, model_name):
    """
    Run 5-fold stratified cross-validation and return fold scores.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1"
    )

    results = {
        "Model": model_name,
        "Fold 1": scores[0],
        "Fold 2": scores[1],
        "Fold 3": scores[2],
        "Fold 4": scores[3],
        "Fold 5": scores[4],
        "Mean F1": scores.mean(),
        "Std F1": scores.std()
    }

    return results


def create_cross_validation_dataframe(
    logistic_model,
    decision_tree_model,
    X,
    y
):
    """
    Run 5-fold cross-validation for both models
    and return a comparison DataFrame.
    """

    logistic_results = run_cross_validation(
        logistic_model,
        X,
        y,
        "Logistic Regression"
    )

    tree_results = run_cross_validation(
        decision_tree_model,
        X,
        y,
        "Decision Tree"
    )

    results_df = pd.DataFrame([
        logistic_results,
        tree_results
    ])

    return results_df