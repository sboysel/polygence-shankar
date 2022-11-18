"""
src/risk_aversion.py

Description: given data on returns, covariance, and asset allocation weights for
a set of stocks, recover the coeviffcient parameterizing investor risk aversion
according to the analytical solution of the Modern Portfolio Theory (MPT)
problem.
"""
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
RISK_AVERSION = pathlib.Path(ROOT, 'data', 'clean', 'risk_aversion.csv')


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

    # find q
    res = minimize_scalar(F, bounds=(0.0, 100.), method='bounded')

    # save result to disk
    q = res.x
    q = pd.DataFrame({'q': [q]})
    q.to_csv(RISK_AVERSION, index=False)

if __name__ == '__main__':
    main()
