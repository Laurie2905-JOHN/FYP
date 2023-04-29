import numpy as np

def moving_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

# Example data with non-constant time steps
time_data = np.array([0, 1, 2.5, 3, 4.5, 5.5, 6, 7, 8.5, 9, 10])
velocity_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# Desired moving average duration (in time units)
moving_average_duration = 3

# Resample the data at a constant time step
time_data_resampled = np.arange(time_data[0], time_data[-1], 0.5)  # Choose a desired constant time step
velocity_data_resampled = np.interp(time_data_resampled, time_data, velocity_data)

# Calculate the window size (number of points) for the moving average
window_size = int(moving_average_duration / (time_data_resampled[1] - time_data_resampled[0]))

# Calculate the moving average of the resampled velocity data
velocity_moving_avg = moving_average(velocity_data_resampled, window_size)

print(velocity_moving_avg)
