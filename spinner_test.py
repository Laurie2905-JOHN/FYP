import numpy as np

# Sample data
data1 = np.array([1, 2, np.nan, 4, 5, 6, 7, np.nan, 9, 10, 11, 12])
data2 = np.array([13, 14, 15, 16, 17, 18, np.nan, 20, 21, 22, 23, 24])
data3 = np.array([13, np.nan, 15, np.nan, 17, 18, np.nan, 20, 21, 22, 23, 24])

def remove_nan_elements(*arrays):
    """Remove elements at the same index in multiple NumPy arrays if any of them contains NaN."""
    # Check for NaN values in each array and combine them
    nan_mask = np.zeros(arrays[0].shape, dtype=bool)
    for array in arrays:
        nan_mask |= np.isnan(array)

    # Remove elements with NaN values from each array
    result = []
    for array in arrays:
        result.append(array[~nan_mask])

    return result


data1_clean, data2_clean, data3_clean = remove_nan_elements(data1, data2, data3)

print("Data 1 (cleaned):")
print(data1_clean)

print("\nData 2 (cleaned):")
print(data2_clean)

print("\nData 3 (cleaned):")
print(data3_clean)