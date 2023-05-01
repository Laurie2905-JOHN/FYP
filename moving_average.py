import numpy as np
import scipy.signal
from scipy.interpolate import interp1d

def moving_average_downsample(data, window_size, downsample_factor):
    # Calculate the moving average using a convolution operation
    kernel = np.ones(window_size) / window_size
    moving_avg = scipy.signal.convolve(data, kernel, mode='valid')

    # Generate the downsampled time indices
    downsampled_indices = np.arange(0, len(moving_avg), downsample_factor)

    # Create an interpolation function based on the moving average data
    interpolation_func = interp1d(np.arange(len(moving_avg)), moving_avg, kind='linear')

    # Use the interpolation function to downsample the moving average data
    downsampled_moving_avg = interpolation_func(downsampled_indices)

    return downsampled_moving_avg

# Generate a signal with 20Hz sampling frequency
sampling_frequency = 20
duration = 10  # seconds
t = np.linspace(0, duration, duration * sampling_frequency)
signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))

# Apply the moving average filter and downsample the signal
window_size = 20
downsample_factor = 10
downsampled_signal = moving_average_downsample(signal, window_size, downsample_factor)

# Print the results
print("Original signal length:", len(signal))
print("Downsampled signal length:", len(downsampled_signal))