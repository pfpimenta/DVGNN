# -*- coding: utf-8 -*-
# generates a box/scatter plot from the results
import matplotlib.pyplot as plt

# creating the dataset

metrics = {}
# static
metrics["static"] = {
    "mae": {"DVGNN+GCN": 4.4807, "DVGNN+MixHop-2": 4.2504, "DVGNN+MixHop-3": 4.2467},
    "rmse": {"DVGNN+GCN": 6.2528, "DVGNN+MixHop-2": 5.9247, "DVGNN+MixHop-3": 5.9461},
    "mape": {
        "DVGNN+GCN": 1120.419,
        "DVGNN+MixHop-2": 938.6775,
        "DVGNN+MixHop-3": 1027.3214,
    },
}
metrics["dynamic"] = {
    "mae": {"DVGNN+GCN": 4.1525, "DVGNN+MixHop-2": 4.1611, "DVGNN+MixHop-3": 4.1343},
    "rmse": {"DVGNN+GCN": 5.8191, "DVGNN+MixHop-2": 5.8061, "DVGNN+MixHop-3": 5.7778},
    "mape": {
        "DVGNN+GCN": 1183.1131,
        "DVGNN+MixHop-2": 1060.539,
        "DVGNN+MixHop-3": 1248.1952,
    },
}

# choose metric to compare
data = metrics["static"]["rmse"]

courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, color="maroon", width=0.4)

# plt.xlabel("DVGNN version")
plt.ylabel("MAE")
# plt.title("asdfg")
plt.show()
