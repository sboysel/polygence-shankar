# polygence shankar

Analyzing the efficacy of structural/analytical economic models (AM) and 
machine learning (ML).

## installation

```sh
> git clone git@github.com:sboysel/polygence-shankar.git
> cd polygence-shankar
> mkdir -p .env
> python -m venv .env/polygence-shankar-py3.10
> source .env/polygence-shankar-py3.10/bin/activate
> pip install -r requirements.txt
```

## main analysis code

Run from the repo directory (e.g., `polygence-shankar`)

```sh
> python src/data.py            # builds dataset
> python src/risk_aversion.py   # recovers risk aversion parameter
> python src/main.py            # main script to evaluate machine learning
                                # models
```

## notebooks

```sh
> jupyter notebook notebooks/main.ipynb
```

## data

[S&P 500 stock data curated by Kaggle user
  camnugent](https://www.kaggle.com/datasets/camnugent/sandp500) (all S&P 500
  stocks, prices and trading volumes):  Sign into Kaggle, download the dataset
  as `.zip`, place archive in `data` subdirectory and extract in place.
```
> cd data
> unzip archive.zip
> ls
total 48M
drwxr-xr-x 4 sam sam 4.0K Nov 17 11:26 individual_stocks_5yr
-rw-r--r-- 1 sam sam  29M Sep 20  2019 all_stocks_5yr.csv
-rw-r--r-- 1 sam sam 4.8K Sep 20  2019 getSandP.py
-rw-r--r-- 1 sam sam  190 Sep 20  2019 merge.sh
-rw-r--r-- 1 sam sam  20M Nov 17 11:24 archive.zip
```

Inspect an individual stock's price history:

```sh
> head individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv 
date,open,high,low,close,volume,Name
2013-02-08,67.7142,68.4014,66.8928,67.8542,158168416,AAPL
2013-02-11,68.0714,69.2771,67.6071,68.5614,129029425,AAPL
2013-02-12,68.5014,68.9114,66.8205,66.8428,151829363,AAPL
2013-02-13,66.7442,67.6628,66.1742,66.7156,118721995,AAPL
2013-02-14,66.3599,67.3771,66.2885,66.6556,88809154,AAPL
2013-02-15,66.9785,67.1656,65.7028,65.7371,97924631,AAPL
2013-02-19,65.8714,66.1042,64.8356,65.7128,108854046,AAPL
2013-02-20,65.3842,65.3842,64.1142,64.1214,118891367,AAPL
2013-02-21,63.7142,64.1671,63.2599,63.7228,111596821,AAPL
```

## references

* 
* https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
