import pickle
import numpy as np
import sys
if sys.platform == "win32":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
with open(r"E:\IBS\dexmachina\arctic\data\arctic_data\data\raw_seqs\s01\box_use_01.mano.npy", "rb") as f:
    hand_mano_data=np.load(f, allow_pickle=True).item()
hand_mano_data["right"]["pose"][0]=[0]*45
hand_mano_data["right"]["trans"][0]=[0]*3
hand_mano_data["right"]["rot"][0]=[0]*3
hand_mano_data["left"]["pose"][0]= [0]*45
hand_mano_data["left"]["trans"][0]=[0]*3 
hand_mano_data["left"]["rot"][0]=[0]*3
