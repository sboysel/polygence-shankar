import numpy as np
import pandas as pd
import pathlib
from scipy.optimize import minimize_scalar

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


def main():
    # read data
    R = pd.read_csv(RETURNS)
    Sigma = pd.read_csv(COV)
    w = pd.read_csv(WEIGHTS)

    # convert to numpy arrays
    R = R['return'].to_numpy()
    Sigma = Sigma.set_index('name').to_numpy()
    w = w['weight'].to_numpy()

    # invert Sigma
    invSigma = np.linalg.inv(Sigma)

    # matrix product of invSigma and returns vector
    invSigmaR = np.dot(invSigma, R)

    # define an objective function to recover risk aversion parameter q
    def F(q: float):
        """
        F is the objective function for the following problem:

        q^* = argmin || w - (0.5 * q * inv(Simga) R) ||

        """
        return np.linalg.norm(w - (0.5 * q) * invSigmaR)

    # print(F(1.0))

    # find q
    res = minimize_scalar(F, bounds=(0.0, 100.), method='bounded')
    # print(res)

    q = res.x
    print(q)

if __name__ == '__main__':
    main()
