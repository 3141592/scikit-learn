import os
import tarfile
import urllib
import urllib.request

print("Download the Data")
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

print("Load the data using pandas")
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
print("housing[\"ocean_proximity\"].value_counts(): ")
print(housing["ocean_proximity"].value_counts())

print("housing.describe(): ")
print(housing.describe())

print("Figure 2.8. A histogram for each numercial attribute")
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.savefig("skl.2.figure.2-8.png")

print("Create a Test Set")
import numpy as np 

housing["income_cat"] = pd.cut(housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5])

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print("strat_test_set")
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

print("Discover and Visualize the Data to Gain Insights")
housing = strat_train_set.copy()

print("Visualizing Geographical Data")
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.savefig("skl.2.figure.2-11.png")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.savefig("skl.2.figure.2-12.png")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing["population"]/100, label="population", figsize=(10, 7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.savefig("skl.2.figure.2-13.png")

print("Looking for Correlations")
corr_matrix = housing.corr()
print("core_matrix[\"median_house_value\"]")
print(corr_matrix["median_house_value"])

print("Figure 2.15. Scatter matrix plot.")
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.savefig("skl.2.figure.2-15.png")



