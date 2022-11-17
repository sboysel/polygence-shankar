"""
src/data.py

Description:
    1. Read historical stock data from Kaggle dataset
    2. Clean
    3. Calculate measures needed for Modern Portfolio Theory analysis: asset
    returns, covariances, and asset allocation weights.
    4. Save cleaned data and measures to `data/clean` 
"""
import glob
import numpy as np
import pandas as pd
import pathlib
import tqdm # progress meter

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
    """
    """
    # read in data from CSV
    d = read()

    # clean/prepare stock data
    d = clean(d)

    # calculate each measure: returns, covariance, and asset allocation weights
    R = measure_returns(d)
    Sigma = measure_covariance(d)
    weight = measure_weights(d)

    # save cleaned data and measures
    d.set_index(['name', 'date']).to_csv(STOCKS)
    R.to_csv(RETURNS)
    Sigma.to_csv(COV)
    weight.to_csv(WEIGHTS)

def read():
    """
    Read in CSV stock histories from `data` subdirectory.  Combine into single
    dataframe.
    """

    # get a list of all CSV filepaths in data subdirectory
    path = str(pathlib.Path(DATA, "*.csv")) # since glob.glob only accepts
                                            # string arguments.
    csv_files = glob.glob(path)

    # list to collect each DataFrame
    dataframes = []

    for filename in tqdm.tqdm(csv_files):
        # read individual file as pd.DataFrame
        df = pd.read_csv(
                filename,
                index_col=None,
                parse_dates=['date']
        )
        # add DataFrame to list
        dataframes.append(df)

    # combine all DataFrames into single DataFrame
    d = pd.concat(dataframes, axis=0, ignore_index=True)
    
    return d

def clean(d: pd.DataFrame):
    """
    Tidy/clean raw data
    """
    # sanitize column names
    d = d.rename(columns={'Name': 'name'})
    
    # reorder columns. places indexes 'name' and 'date' on the left.  dropping
    # 'high', 'low', and 'close' since we don't need them.  notice we could have
    # also used 'close' in place of 'open'.
    d = d[['name', 'date', 'open', 'volume']]

    # rename 'open' to 'price' for clarity
    d = d.rename(columns={'open': 'price'})

    # sort by stock ticker name (ascending alphabetical) and then date
    # (ascending)
    d = d.sort_values(['name', 'date'])

    return d

"""
Functions to calculate measures needed for Modern Portfolio Theory model.
"""


def measure_returns(d: pd.DataFrame):
    """
    Calculate the return for the set of S&P stocks.

    Assumptions:
        1. The return is calculated as buying the stock at the initial sample
        period (i.e., 2013-02-08) and selling at the final observation period
        (i.e., 2018-02-07)
    """
    # first price
    price_first = d.rename(columns={'price': 'price_first'}) \
            .groupby('name')['price_first'] \
            .first()

    # last price
    price_last = d.rename(columns={'price': 'price_last'}) \
            .groupby('name')['price_last'] \
            .last()

    # return = (price_last - price_first) / price_first
    r = pd.concat([price_first, price_last], axis=1)
    r['return'] = (r['price_last'] - r['price_first']) / r['price_first']
    
    # tidy up return object
    r = r['return']

    return r


def measure_covariance(d: pd.DataFrame):
    """
    Calculate variance-covariance matrix between individual stocks in S&P 500.

    Reference: https://en.wikipedia.org/wiki/Covariance_matrix
    """
    # pivot data from long to wide format.  each column is now a sequence of
    # prices for each individual stock
    d_wide = pd.pivot(d, index='date', columns='name', values='price')

    # calculate variance-covariance matrix.  return object is pd.DataFrame
    sigma = d_wide.cov()

    return sigma

def measure_weights(d: pd.DataFrame):
    """
    Calculate average asset allocation weights.

    Assumptions:
        1. Use average trading volumes to proxy for shares outstanding over
        sample period.
    """
    # get aggregate trading volume by date
    volume_t = d.groupby('date')[['volume']] \
            .sum() \
            .rename(columns={'volume': 'volume_t'})
    d = d.set_index('date').join(volume_t)

    # for each day, calculate each stock's share of trading volume 
    d['weight'] = d['volume'] / d['volume_t']

    # take an average a stock's wieght across days in the sample period
    weight = d.reset_index().groupby('name')['weight'].mean()

    # print(weight)
    # print(weight.sum()) = 1.024064371178093

    return weight

if __name__ == '__main__':
    main()
