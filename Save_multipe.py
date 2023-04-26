import os
import numpy as np

# Create NumPy arrays
data1 = np.random.random((10000, 10000))
data2 = np.random.random((500, 500))

# Specify the folder and file name
folder_path = r'C:\Users\lauri\OneDrive\Documents (1)\University\Year 3\Semester 2\BARNACLE\Example Data\Workspace'
file_name = 'data.npz'

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save the NumPy arrays to the specified folder
# file_path = os.path.join(folder_path, file_name)
#
# # Save multiple arrays to the same file (uncompressed)
# np.savez(file_path, data1=data1, data2=data2)

# Save multiple arrays to the same file (compressed)
# np.savez_compressed(file_path, data1=data1, data2=data2)

# Load the arrays from the file

folder_path = r'C:\Users\lauri\OneDrive\Documents (1)\University\Year 3\Semester 2\BARNACLE\Example Data\Workspace'
file_name = 'Example 1.npz'
file_path = os.path.join(folder_path, file_name)

# loaded_data = np.load(file_path)
#
# print(loaded_data['Mon1527'])

import matplotlib.pyplot as plt

x = np.load(file_path)
print(x)
for k in x.keys():
    print(k)

fig, axs = plt.subplots(4, 1)
axs[0].plot(x['t'], x['Ux'], 'k')
axs[1].plot(x['t'], x['Uy'], 'r')
axs[2].plot(x['t'], x ['Uz'], 'r')
axs[3].plot(x['t'], x['U1'], 'r')

plt.show()