import numpy as np

# Your time series data (unequally spaced)
timestamps = np.array([0, 1.2, 2.1, 3.2, 4.5, 5.6, 6.7, 7.8, 8.9, 10])
data = np.array([1, 2, 3, 2, 3, 4, 5, 4, 5, 6])

# Desired moving average duration (in seconds)
duration = 3

# Sample frequency (SF) - data points per second
SF = 1

# Calculate the window size based on the desired duration and sample frequency
window_size = int(duration * SF)

# Resample the data to equally spaced time series using linear interpolation
new_timestamps = np.arange(timestamps[0], timestamps[-1], 1 / SF)
resampled_data = np.interp(new_timestamps, timestamps, data)

# Create a window for the moving average calculation
window = np.ones(window_size) / window_size

# Calculate the moving average using numpy's convolve function
moving_average = np.convolve(resampled_data, window, mode='valid')

# Print the moving average
print(moving_average)
