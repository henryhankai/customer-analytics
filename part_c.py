import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functions import *
from variables import *

# load data
df = pd.read_csv("data/mugs-preference-parameters-full.csv", index_col=0)
df.rename(columns=str.strip, inplace=True)

# Initialise X dataframe with importances
X = df[[name for name in df.columns if name.startswith("I")]].copy()

# Get all preference characteristics
for param in [name for name in df.columns if name.startswith("p")]:
    if "In" in param:
        X["I*" + param] = df["Iin"] * df[param]
    elif "Cn" in param:
        X["I*" + param] = df["Icn"] * df[param]
    else:
        X["I*" + param] = df["I" + str(param[1:3])] * df[param]

# X is an np.array containing the data
k = 5 # change to desired
random.seed(410014)
kmeansModel = KMeans(n_clusters=k, n_init=50, max_iter=100)
kmeansModel.fit(X)
labels = kmeansModel.labels_

# Identify segments
X["segment"] = kmeansModel.labels_
z = X.groupby("segment").mean()

for i in range(len(labels)):
    if (z.loc[labels[i], "IPr"] < 13):
        labels[i] = 3
    elif (z.loc[labels[i], "IPr"] < 14):
        labels[i] = 1
    elif (z.loc[labels[i], "IPr"] < 15.5):
        labels[i] = 2
    elif (z.loc[labels[i], "IPr"] < 17):
        labels[i] = 4
    else:
        labels[i] = 5

X["segment"] = kmeansModel.labels_
print(X.groupby("segment").mean())

# PCA code
from sklearn.decomposition import PCA
pca_2 = PCA(2)
# Standardize the X matrix to make PCA operate on correlations instead of covariances
from numpy import mean, std
X_std = (X - mean(X)) / std(X)
plot_columns = pca_2.fit_transform(X_std)
scatter = plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.legend(*scatter.legend_elements(), title="segment number")
pd.DataFrame(pca_2.components_, columns=X.columns).to_csv('pca.csv')

plt.show()
