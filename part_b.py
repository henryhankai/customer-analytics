import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functions import *
from variables import *

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

ave_within_cluster_mean_sum_of_squares = []
for k in range(2, 11):
    random.seed(410014)
    kmeansModel = KMeans(n_clusters=k, n_init=50, max_iter=100)
    kmeansModel.fit(X)
    ave_within_cluster_mean_sum_of_squares.append(kmeansModel.inertia_ / X.shape[0])

plt.plot(range(2,11), ave_within_cluster_mean_sum_of_squares)
plt.xlabel('k')
plt.ylabel('Average Within Group Variance')
plt.show()

k = 5 # change to desired
random.seed(410014)
kmeansModel = KMeans(n_clusters=k, n_init=50, max_iter=100)
kmeansModel.fit(X)
X["segment"] = kmeansModel.labels_

# Append demographics
demographics_df = pd.read_csv("data/demographics-full.csv", index_col=0)
X = pd.concat([X, demographics_df], axis=1)

out = X.groupby("segment").mean()
loglift = np.log10(out.to_numpy() / X.drop(["segment"], axis=1).mean(axis=0).to_numpy()[np.newaxis, :])
out = out.append((X.drop(["segment"], axis=1).mean(axis=0)),ignore_index=True)

print(out)
print(pd.DataFrame(loglift, columns=out.columns))
out.to_csv('part_b_segment_profiles.csv')
pd.DataFrame(loglift, columns=out.columns).to_csv('part_b_log_lift.csv')
