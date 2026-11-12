"""
Boston Housing Linear Regression Project
Loads the historical Boston Housing dataset from OpenML, trains a
scikit-learn LinearRegression model, evaluates it, and saves a plot.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("Downloading/loading the Boston Housing dataset from OpenML...")
    # The Boston dataset was removed from scikit-learn's built-in datasets.
    # OpenML provides the historical dataset; an internet connection is required
    # the first time it is downloaded.
    boston = fetch_openml(name="boston", version=1, as_frame=True)

    X = boston.data
    y = pd.to_numeric(boston.target)

    print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict house prices for the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel evaluation")
    print(f"MAE : {mae:.3f}")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # Plot actual versus predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.65)
    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())
    plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
    plt.xlabel("Actual house value")
    plt.ylabel("Predicted house value")
    plt.title("Boston Housing: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_plot.png", dpi=300)
    print("\nSaved chart as prediction_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
