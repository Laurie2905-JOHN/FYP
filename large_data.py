import os
import numpy as np

# Set your custom file path here
file_path = r""

# Read the input file and create a NumPy array
with open(os.path.join(file_path, "Example 1.txt"), "r") as file:
    # Load the data from the file using ',' as delimiter and store it in a NumPy array
    data_array = np.loadtxt(file, delimiter=',')

# Set your sample rate (fs) and the time threshold here
fs = 16  # Example sample rate

# Calculate the Time array
# Create an evenly spaced array of time values with the same length as the data array
Time = np.linspace(0, len(data_array) / fs, len(data_array))

# Calculate the total time duration of the data
total_time = Time[-1]

# Set the desired time duration
desired_time = 31*24*60*60  # Example desired time duration in seconds

# Calculate how many times the data should be repeated to fill the desired time
repeat_count = int(np.ceil(desired_time / total_time))

# Repeat the data to fill the desired time
# Tile the data_array 'repeat_count' times along the first axis
extended_data = np.tile(data_array, (repeat_count, 1))

# Save the extended data as a new text file
with open(os.path.join(file_path, "outputLONG_SF16.txt"), "w") as file:
    # Save the extended data to the output file using ',' as delimiter and formatted as strings
    np.savetxt(file, extended_data, delimiter=',', fmt='%s')