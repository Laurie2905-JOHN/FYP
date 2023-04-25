
import os
import numpy as np

# Set your custom file path here
file_path = r"C:\Users\lauri\OneDrive\Documents (1)\University\Year 3\Semester 2\BARNACLE\Example Data\BARNACLE DATA"

# Read the input file and create a NumPy array
with open(os.path.join(file_path, "Example 1.txt"), "r") as file:
    data_array = np.loadtxt(file, delimiter=',')

# Set your sample rate (fs) and the time threshold here
fs = 16  # Example sample rate

# Calculate the Time array
Time = np.linspace(0, len(data_array) / fs, len(data_array))

# Calculate the total time duration of the data
total_time = Time[-1]

# Set the desired time duration
desired_time = 31*24*60*60

# Example desired time duration in seconds

# Calculate how many times the data should be repeated to fill the desired time
repeat_count = int(np.ceil(desired_time / total_time))

# Repeat the data to fill the desired time
extended_data = np.tile(data_array, (repeat_count, 1))

# Save the extended data as a new text file
with open(os.path.join(file_path, "output.txt"), "w") as file:
    np.savetxt(file, extended_data, delimiter=',', fmt='%s')

