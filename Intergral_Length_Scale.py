import numpy as np

def autocorrelation_function(velocity_component):
    n = len(velocity_component)
    mean = np.mean(velocity_component)
    autocorr = np.correlate(velocity_component - mean, velocity_component - mean, mode='full')[-n:]
    autocorr /= autocorr[0]
    return autocorr

def integral_length_scale(velocity_component, delta):
    autocorr = autocorrelation_function(velocity_component)
    integral_scale = np.trapz(autocorr, dx=delta)
    return integral_scale

# Example data (replace with your actual data)
x_velocity = np.random.random(1000)
y_velocity = np.random.random(1000)
z_velocity = np.random.random(1000)

# The distance between data points (replace with your actual delta value)
delta = 0.01

# Calculate the integral length scales
x_integral_scale = integral_length_scale(x_velocity, delta)
y_integral_scale = integral_length_scale(y_velocity, delta)
z_integral_scale = integral_length_scale(z_velocity, delta)

print(f"Integral length scale for x: {x_integral_scale}")
print(f"Integral length scale for y: {y_integral_scale}")
print(f"Integral length scale for z: {z_integral_scale}")