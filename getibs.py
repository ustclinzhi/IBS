import os
import torch
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from viewibs import DifferentiableIBS

def save_points_as_obj(points_tensor, filename):
    """
    Saves a point cloud tensor (1, N, 3) to an OBJ file.
    """
    points_np = points_tensor.squeeze(0).cpu().numpy()
    with open(filename, 'w') as f:
        for p in points_np:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
    print(f"Saved {filename}")

def process_folder(root_folder):
    """
    Traverses subfolders, extracts OBJ files, calculates IBS, and saves results.
    """
    all_ibs_point_clouds_left = []
    all_ibs_point_clouds_right = []
    all_obj_point_clouds=[]
    
    # Initialize DifferentiableIBS outside the loop if it's computationally expensive to re-initialize
    ibs_calculator = DifferentiableIBS(max_iter=60, n_samples=4096)
    
    # Check for CUDA availability and set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        ibs_calculator.to(device)
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device('cpu')
        ibs_calculator.to(device)
        print("CUDA not available, using CPU.")

    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        if os.path.isdir(subfolder_path):
            print(f"\nProcessing subfolder: {subfolder_name}")
            obj_files = [f for f in os.listdir(subfolder_path) if f.endswith('.obj')]

            if len(obj_files) >= 2:
                # Assuming the first two OBJ files found are the ones to be used
                obj_file1_path = os.path.join(subfolder_path, "right_hand.obj")
                obj_file2_path = os.path.join(subfolder_path, "object.obj")
                obj_file3_path=os.path.join(subfolder_path, "left_hand.obj")

                try:
                    # Load OBJ files
                    verts1, faces1, _ = load_obj(obj_file1_path)
                    verts2, faces2, _ = load_obj(obj_file2_path)
                    verts3,_,_=load_obj(obj_file3_path)

                    
                    print(f"Point cloud 1 shape: {verts1.shape}")
                    print(f"Point cloud 2 shape: { verts2.shape}")

                    #Calculate IBS point cloud
                    ibs_point_cloudright = ibs_calculator(verts2.unsqueeze(0), verts1.unsqueeze(0))
                    ibs_point_cloudleft = ibs_calculator(verts2.unsqueeze(0), verts3.unsqueeze(0))
                    print(f"Calculated IBS point cloud shape: {ibs_point_cloudleft.shape}")

                    #Save IBS point cloud as OBJ
                    ibs_obj_filename = os.path.join(subfolder_path, "ibsright.obj")
                    save_points_as_obj(ibs_point_cloudright , ibs_obj_filename)
                    ibs_obj_filename = os.path.join(subfolder_path, "ibsleft.obj")
                    save_points_as_obj(ibs_point_cloudleft , ibs_obj_filename)

                    #Append to the list of all IBS point clouds
                    all_obj_point_clouds.append(verts2.unsqueeze(0).cpu().numpy())
                    all_ibs_point_clouds_left.append(ibs_point_cloudleft.cpu().numpy())
                    all_ibs_point_clouds_right.append(ibs_point_cloudright.cpu().numpy())

                except Exception as e:
                    print(f"Error processing {subfolder_name}: {e}")
            else:
                print(f"Skipping {subfolder_name}: Not enough OBJ files found (expected at least 2).")

    # Save all IBS point clouds into a single .npy file
    if all_ibs_point_clouds_left:
        final_npy_path1 = os.path.join(root_folder, "demoibsright.npy")
        # Stack all (1, 4096, 3) arrays into a (num_subfolders, 4096, 3) array
        np.save(final_npy_path1, np.vstack(all_ibs_point_clouds_right))
        final_npy_path2 = os.path.join(root_folder, "demoibsleft.npy")
        # Stack all (1, 4096, 3) arrays into a (num_subfolders, 4096, 3) array
        np.save(final_npy_path2, np.vstack(all_ibs_point_clouds_left))
        final_npy_path3 = os.path.join(root_folder, "demoobj.npy")
        # Stack all (1, 4096, 3) arrays into a (num_subfolders, 4096, 3) array
        np.save(final_npy_path3, np.vstack(all_obj_point_clouds))
        print(f"\nAll IBS point clouds saved ")
    else:
        print("\nNo IBS point clouds were generated.")

if __name__ == "__main__":
    # --- IMPORTANT: Set your root folder here ---
    # Replace 'your_root_folder_path' with the actual path to your main folder
    # containing the subfolders with OBJ files.
    # Example: root_folder_to_process = "/Users/your_username/data/my_3d_models"
    root_folder_to_process = "/mnt/e/IBS/dexmachina/arctic/output_meshes_mixer_use_01" 

    if root_folder_to_process == "your_root_folder_path":
        print("------------------------------------------------------------")
        print("WARNING: Please set 'root_folder_to_process' to your actual data directory.")
        print("Example: root_folder_to_process = '/path/to/your/main_folder'")
        print("------------------------------------------------------------")
    else:
        # Check if the root folder exists
        if os.path.isdir(root_folder_to_process):
            process_folder(root_folder_to_process)
        else:
            print(f"Error: The specified root folder does not exist: {root_folder_to_process}")