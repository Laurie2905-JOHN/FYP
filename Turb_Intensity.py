import numpy as np

def calculate_turbulence_intensity(u, v, w):
    N = len(u)

    # Calculate mean velocities
    U = np.mean(u)
    V = np.mean(v)
    W = np.mean(w)

    # Calculate velocity fluctuations
    u_prime = u - U
    v_prime = v - V
    w_prime = w - W

    # Calculate mean squared velocity fluctuations
    mean_u_prime_sq = np.mean(np.square(u_prime))
    mean_v_prime_sq = np.mean(np.square(v_prime))
    mean_w_prime_sq = np.mean(np.square(w_prime))

    # Calculate RMS of velocity fluctuations
    u_prime_RMS = np.sqrt(mean_u_prime_sq)
    v_prime_RMS = np.sqrt(mean_v_prime_sq)
    w_prime_RMS = np.sqrt(mean_w_prime_sq)

    # Calculate magnitude of mean flow velocity
    U_mag = np.sqrt(U**2 + V**2 + W**2)

    # Calculate turbulence intensity
    TI = (np.sqrt(u_prime_RMS**2 + v_prime_RMS**2 + w_prime_RMS**2)) / U_mag

    return TI

# Example usage:
u = np.array([1.0, 1.1, 1.2, 0.9, 1.0, 1.1])
v = np.array([0.5, 0.6, 0.4, 0.5, 0.5, 0.6])
w = np.array([0.3, 0.4, 0.3, 0.2, 0.3, 0.4])

TI = calculate_turbulence_intensity(u, v, w)
print("Turbulence Intensity:", TI)