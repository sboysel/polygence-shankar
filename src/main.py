"""
src/main.py

Description: Compare machine learning methods to predict optimal portfolio
weights. 
"""
import numpy as np
import pandas as pd
import pathlib
from sklearn import linear_model

# constants
ROOT = pathlib.Path.cwd()
DATA = pathlib.Path(
        ROOT,
        'data',
        'raw',
        'individual_stocks_5yr', 
        'individual_stocks_5yr'
)
STOCKS = pathlib.Path(ROOT, 'data', 'clean', 'stocks.csv')
RETURNS = pathlib.Path(ROOT, 'data', 'clean', 'returns.csv')
COV = pathlib.Path(ROOT, 'data', 'clean', 'covariance.csv')
WEIGHTS = pathlib.Path(ROOT, 'data', 'clean', 'weight.csv')
RISK_AVERSION = pathlib.Path(ROOT, 'data', 'clean', 'risk_aversion.csv')


def main():
    # read data
    R = pd.read_csv(RETURNS)
    Sigma = pd.read_csv(COV)
    w = pd.read_csv(WEIGHTS)
    q = pd.read_csv(RISK_AVERSION)

    # convert to numpy arrays
    R = R['return'].to_numpy()
    Sigma = Sigma.set_index('name').to_numpy()
    w = w['weight'].to_numpy()
    q = q.q

    """
    linear regression
    """
    n = len(w)
    y = w
    X = np.concatenate((R.reshape((-1, 1)), Sigma), axis=1)
   
    # 90% test training split
    n90 = int(n * 0.9)
    X_train, y_train = X[:n90,], y[:n90]
    X_test, y_test = X[n90:,], y[n90:]

    # baseline model: w = g(Sigma, R)
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)

    # evaluate fit
    print("Baseline linear regression (R squared): ", reg.score(X_test, y_test))

    """
    LASSO regression
    """
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(X_train, y_train)

    # evaluate fit
    print("Lasso regression, alpha=0.1 (R squared): ", reg.score(X_test, y_test))

    reg = linear_model.Lasso(alpha=0.2)
    reg.fit(X_train, y_train)

    # evaluate fit
    print("Lasso regression, alpha=0.2 (R squared): ", reg.score(X_test, y_test))

    reg = linear_model.Lasso(alpha=0.5)
    reg.fit(X_train, y_train)

    # evaluate fit
    print("Lasso regression, alpha=0.5 (R squared): ", reg.score(X_test, y_test))



if __name__ == '__main__':
    main()
