import os
from stl import mesh

def scale_stl_files(directory, scale_factor=0.001):
    """
    Scale all STL files in the given directory by the specified scale factor.

    :param directory: Path to the directory containing STL files.
    :param scale_factor: Scaling factor to apply to the STL files.
    """
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    stl_files = [f for f in os.listdir(directory) if f.endswith('.stl')]
    if not stl_files:
        print(f"No STL files found in directory: {directory}")
        return

    for stl_file in stl_files:
        file_path = os.path.join(directory, stl_file)
        print(f"Processing file: {file_path}")

        # Load the STL file
        try:
            stl_mesh = mesh.Mesh.from_file(file_path)
        except Exception as e:
            print(f"Failed to load STL file {file_path}: {e}")
            continue

        # Scale the vertices
        stl_mesh.vectors *= scale_factor

        # Save the scaled STL file (overwrite the original file)
        try:
            stl_mesh.save(file_path)
            print(f"Scaled and saved: {file_path}")
        except Exception as e:
            print(f"Failed to save scaled STL file {file_path}: {e}")

if __name__ == "__main__":
    # Path to the directory containing STL files
    stl_directory = r"E:\IBS\dexmachina\dexmachina\assets\new_hand\right\meshes"

    # Scale factor to convert mm to m
    scale_factor = 0.001

    # Scale all STL files in the directory
    scale_stl_files(stl_directory, scale_factor)