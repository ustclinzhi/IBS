

import numpy as np

# Load the data from the .npy file
filepath = r"E:\IBS\dexmachina\arctic\output_meshes\demoobj.npy"  # Replace with the path to your file
data = np.load(filepath)

compressed_data = data[:, ::3, :]


# Save the compressed data to a new .npy file
output_filepath = r"E:\IBS\dexmachina\arctic\output_meshes\smalldemoobj.npy"
np.save(output_filepath, compressed_data)

print(f"Original data shape: {data.shape}")
print(f"Compressed data shape: {compressed_data.shape}")
print(f"Data saved to {output_filepath}")