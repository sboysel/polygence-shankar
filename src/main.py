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
    n = len(y)
    y = w
    X = np.concatenate((Sigma, R.reshape((-1, 1))), axis=1)
    
    # baseline model: w = g(Sigma, R)
    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    # evaluate fit
    print("Baseline linear regression (R squared): ", reg.score(X, y))

    """
    LASSO regression
    """
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(X, y)

    # evaluate fit
    print("Lasso regression, alpha=0.1 (R squared): ", reg.score(X, y))


if __name__ == '__main__':
    main()
