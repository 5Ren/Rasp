import csv
import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pharos_simu', 'simulation_csv', 'simu.csv')

with open(path) as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    data = [row for row in reader]
    data = np.array(data[1:][:])
print(data)

x = data[:, 0]
y = data[:, 1]
plt.axes().set_aspect('equal')
plt.grid()
plt.plot(x, y, marker="v", color = "blue", linestyle = ":")
plt.show()

