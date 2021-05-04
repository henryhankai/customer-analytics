import numpy as np

# Parameters
c = 100/(12*600)

# Possible features and their values
product_features = np.array(["Pr", "In", "Cp", "Cl", "Cn", "Br"])
price = np.array(["30","10","05"])
time_insulated = np.array(["0.5", "1", "3"])
capacity = np.array(["12", "20", "32"])
cleanability = np.array(["D", "F", "E"])
cleanability_full_names = {
    "D": "Difficult",
    "F": "Fair",
    "E": "Easy"
}
containment = np.array(["Sl", "Sp", "Lk"])
containment_full_names ={
    "Sl": "Slosh resistant",
    "Sp": "Spill resistant",
    "Lk": "Leak resistant"
}
brand = np.array(["A", "B", "C"])

# Incumbents data
incumbent_1 = ("30", "3", "20", "E", "Lk", "A")
incumbent_2 = ("10", "1", "20", "F", "Sp", "B")
proposed_candidate = ("30", "3", "20", "E", "Lk", "C")
products = (incumbent_1, incumbent_2, proposed_candidate)

# Costs data
time_insulated_costs = {
    "0.5": 0.5,
    "1": 1,
    "3": 3
}
capacity_costs = {
    "12": 1,
    "20": 2.6,
    "32": 2.8
}
cleanability_costs = {
    "D": 1,
    "F": 2.2,
    "E": 3
}
containment_costs = {
    "Sl": 0.5,
    "Sp": 0.8,
    "Lk": 1
}
cost_dict = {
    "time insulated": time_insulated_costs,
    "capacity": capacity_costs,
    "cleanability": cleanability_costs,
    "containment": containment_costs
}
