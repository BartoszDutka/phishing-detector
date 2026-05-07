from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def train_models(X_train, y_train):
    """
    Train Logistic Regression and Random Forest models.

    Returns:
        dict: {'Logistic Regression': model1, 'Random Forest': model2}
    """
    print("Training Logistic Regression...")
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=42))
    ])
    logistic_model.fit(X_train, y_train)

    print("Training Random Forest...")
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    random_forest_model.fit(X_train, y_train)

    print("Models trained successfully.")

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }
