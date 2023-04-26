import numpy as np

import pandas as pd

# Create three sample arrays
e = np.array([['t','d']])

a = np.array([0.18689561, 0.17042526, 0.19090303, 0.07513609, 0.05079986,
       0.06968498])

b = np.array([0.18689561, 0.17042526, 0.19090303, 0.07513609, 0.05079986,
       0.06968498])

# Concatenate the arrays vertically
concatenated_array = np.column_stack([a,b])

concatenated_array1 = np.concatenate((e,concatenated_array))

print(concatenated_array1)

# Save the concatenated array as a CSV file
np.savetxt(r"C:\Users\lauri\OneDrive\Documents (1)\University\Year 3\Semester 2\BARNACLE\Example Data\Workspace\fi22uu33dd3l133e.csv", concatenated_array1, delimiter=",", fmt="%s")

# # Save the concatenated array as an Excel file
# df = pd.DataFrame(concatenated_array1)
# df.to_excel(r"C:\Users\lauri\OneDrive\Documents (1)\University\Year 3\Semester 2\BARNACLE\Example Data\Workspace\f13le.xlsx", index=False, header=False)