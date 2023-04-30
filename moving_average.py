import numpy as np
import scipy

# Define the moving average function
def moving_average(data, window_size):
    # Create an array (kernel) of ones with the shape of window_size
    # and normalize it by dividing each element by the window_size.
    # This kernel will be used to compute the moving average.
    kernel = np.ones(window_size) / window_size

    # Apply the kernel to the input data using convolution. This operation
    # computes the moving average by sliding the kernel over the data and
    # calculating the dot product between the kernel and the data in the
    # current window. The 'valid' mode ensures that the output array size
    # is reduced to only include positions where the kernel and data fully overlap.
    return scipy.signal.fftconvolve(data, kernel, mode = 'valid')

# Resample the data at a constant time step
time_data_resampled = np.arange(time_data[0], time_data[-1], 0.5)  # Choose a desired constant time step
velocity_data_resampled = np.interp(time_data_resampled, time_data, velocity_data)

# Calculate the window size (number of points) for the moving average
window_size = int(moving_average_duration / (time_data_resampled[1] - time_data_resampled[0]))

# Calculate the moving average of the resampled velocity data
velocity_moving_avg = moving_average(velocity_data_resampled, window_size)

print(velocity_moving_avg)
