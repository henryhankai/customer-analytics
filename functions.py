import numpy as np
import pandas as pd

def calculate_utility(df, features):
    pr, ins, cp, cl, cn, brand = features
    return  df["IPr"] * df["pPr" + str(pr)] + \
            df["Iin"] * df["pIn" + str(ins)] + \
            df["ICp"] * df["pCp" + str(cp)] + \
            df["ICl"] * df["pCl" + str(cl)] + \
            df["Icn"] * df["pCn" + str(cn)] + \
            df["IBr"] * df["pBr" + str(brand)]

def calculate_cost(cost_dict, features):
    pr, ins, cp, cl, cn, brand = features
    return  cost_dict["time insulated"][ins] + \
            cost_dict["capacity"][cp] + \
            cost_dict["cleanability"][cl] + \
            cost_dict["containment"][cn]

def get_consumer_preference(row, num_features):
    return (row.to_numpy().reshape(-1, num_features).T)
