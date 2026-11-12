# Boston Housing – Linear Regression

A beginner-friendly machine-learning project that trains and evaluates a
Linear Regression model using scikit-learn on the historical Boston Housing
dataset.

## Project steps

1. Import the required Python libraries.
2. Load the Boston Housing dataset from OpenML.
3. Split the data into training (80%) and testing (20%) sets.
4. Train a `LinearRegression` model.
5. Evaluate predictions using MAE, MSE, RMSE, and R².
6. Plot actual versus predicted house values.

## Requirements

- Python 3.9 or later
- Internet connection for the first dataset download

## Setup

Open a terminal in this project folder and run:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python linear_regression.py
```

The script prints evaluation metrics and saves `prediction_plot.png` in the
project folder.

## Dataset note

The original Boston Housing dataset was removed from scikit-learn because of
ethical concerns and is not included in current scikit-learn installations.
This educational task uses the historical dataset hosted on OpenML instead.
The dataset is downloaded when the script runs, so an internet connection is
needed. Consider using a different dataset for new real-world projects.

## Files

- `linear_regression.py` – complete model implementation
- `requirements.txt` – Python dependencies
- `prediction_plot.png` – generated automatically when the script runs
