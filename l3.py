import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

#import data
df = pd.read_csv("data\\housing.csv")
print(df.isna().any(axis=0).sum())   #how many columns have missing values
print(df.isna().any(axis=1))  #returns rows same sized object where if the object is missing it ouput true, else false
df=df.dropna()
df = df.reset_index()

from sklearn.preprocessing import OneHotEncoder
my_encoder = OneHotEncoder()
encoded_data = my_encoder.transform(df[['ocean_proximity']])
category_names= my_encoder.get_feature_names_out()
encoded_data_df = pd.DataFrame(encoded_data, columns= category_names)
df = pd.concat([df, encoded_data_df], axis = 1)
df = df.drop('ocean_proximity')




#stratified sampling
df["income_cat"] = pd.cut(df["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
for a in (strat_train_set, strat_test_set):
    a.drop("income_cat", axis=1)

#anything from here, unless for testing, should be on only the train dataset
#to avoid data snooping bias.
train_y = strat_test_set['median_house_value']
train_x = strat_test_set.drop(columns='median_house_value')
scaled_data = my_scaler.transform(train_x)


from sklearn.preprocessing import StandardScaler
my_scaler = StandardScaler()
my_scaler.dit(strat_train_set)





#creating scatter plots using panda
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
pd.plotting.scatter_matrix(df[attributes], figsize=(12, 8))

#looking at correlations
corr_matrix = strat_train_set.corr(numeric_only=True)
plt.figure()
# plt.matshow(corr_matrix)
sns.heatmap(np.abs(corr_matrix)); #this generates a better looking correlations
                                  #matrix compared to plt.matshow()

selected_variables = ['longitude', 'housing_median_age', 'total_rooms',
                     'median_income','ocean_proximity']

strat_train_set_selected = strat_train_set[selected_variables]