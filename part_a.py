import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from variables import *

###############################################################################
# Test functions
# def calculate_utility(df, features):
#     pr, cp, cl, brand = features
#     return  df["IPr"] * df["pPr" + str(pr)] + \
#             df["ICp"] * df["pCp" + str(cp)] + \
#             df["ICl"] * df["pCl" + str(cl)] + \
#             df["IBr"] * df["pBr" + str(brand)]
# incumbent_1 = ("30", "20", "E", "A")
# incumbent_2 = ("10", "20", "F", "B")
# proposed_candidate = ("30", "20", "E", "C")
# products = (incumbent_1, incumbent_2, proposed_candidate)
# df = pd.read_csv("pref_test.csv", index_col=0)
# demographics_df = pd.read_csv("demo_test.csv", index_col=0)
# ###############################################################################

df = pd.read_csv("data/mugs-preference-parameters-full.csv", index_col=0)
df.rename(columns=str.strip, inplace=True)

# get utilites
for i, product in enumerate(products):
    df["U_" + str(i+1)] = calculate_utility(df, product)
    df["ecU_" + str(i+1)] = np.exp(c * df["U_" + str(i+1)])

df["sum_ecU"] = df[[name for name in df.columns if name.startswith("ecU_")]].sum(axis=1)

# get probabilities
for i, product in enumerate(products):
    df["P_" + str(i+1)] = df["ecU_" + str(i+1)] / df["sum_ecU"]

# Initialise Affinity based segment dataframe with importances
affinity_seg_df = df[[name for name in df.columns if name.startswith("I")]].copy()

# Get all preference characteristics
for param in [name for name in df.columns if name.startswith("p")]:
    if "In" in param:
        affinity_seg_df["I*" + param] = df["Iin"] * df[param]
    elif "Cn" in param:
        affinity_seg_df["I*" + param] = df["Icn"] * df[param]
    else:
        affinity_seg_df["I*" + param] = df["I" + str(param[1:3])] * df[param]

# Append demographics
demographics_df = pd.read_csv("data/demographics-full.csv", index_col=0)
affinity_seg_df = pd.concat([affinity_seg_df, demographics_df], axis=1)
affinity_seg_df.to_csv('affinity_seg.csv', index_label = "Cust")

out = []
for i, product in enumerate(products):
    df_temp = affinity_seg_df.multiply(df["P_" + str(i+1)], axis=0)
    df_temp = df_temp.sum(axis=0) / df["P_" + str(i+1)].sum(axis=0)
    out.append(df_temp.to_numpy())

loglift = np.log10(out / affinity_seg_df.mean(axis=0).to_numpy()[np.newaxis, :])
print(pd.DataFrame(loglift, columns=affinity_seg_df.columns))

out.append(affinity_seg_df.mean(axis=0).to_numpy())
print(pd.DataFrame(out, columns=affinity_seg_df.columns))
pd.DataFrame(out, columns=affinity_seg_df.columns).to_csv('part_a_segment_profiles.csv')
pd.DataFrame(loglift, columns=affinity_seg_df.columns).to_csv('part_a_log_lift.csv')
