from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classification model.
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classification model.
    """
    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    return model