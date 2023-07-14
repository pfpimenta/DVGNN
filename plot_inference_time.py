# -*- coding: utf-8 -*-
# generates a box/scatter plot from the results
import matplotlib.pyplot as plt

# creating the dataset
data = {"DVGNN+GCN": 10.6, "DVGNN+MixHop-2": 33.8, "DVGNN+MixHop-3": 47.6}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, color="maroon", width=0.4)

# plt.xlabel("DVGNN version")
plt.ylabel("Average time per epoch in seconds")
# plt.title("asdfg")
plt.show()
