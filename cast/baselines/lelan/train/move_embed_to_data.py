import os 
import glob
import shutil 
import numpy as np

paths = glob.glob("/home/cglossop/atomic_dataset/*/*/*/traj_data.pkl", recursive=True)
breakpoint()

for path in paths:

    data = np.load(path, allow_pickle=True)

    if "language_instruction" not in data.keys() or "varied_language_instruction" not in data.keys():
        print(path)
        breakpoint()