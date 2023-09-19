import numpy as np
import pandas as pd
import matplotlib as plt

df = pd.read_csv("data/housing.csv")

print(df.info())

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=3224)

df["income_cat"] = pd.cut(df["median_income"], bins=[0.,1.5,3.0,4.5,6.])


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]