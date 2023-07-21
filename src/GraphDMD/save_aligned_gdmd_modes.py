import numpy as np
import os
from src.CONSTANTS import CONSTANTS
import argparse

HOME = CONSTANTS.HOME

subject_ids = list(np.loadtxt(os.path.join(HOME, "subjectIDs.txt"), dtype=int))
filtered_subject_ids = np.loadtxt(os.path.join(CONSTANTS.CODEDIR, "src/GraphDMD", "filteredSubjectIds.txt"))
parser = argparse.ArgumentParser()
parser.add_argument("--min_psi", type=float)
parser.add_argument("--max_psi", type=float)
args = parser.parse_args()

min_psi = args.min_psi
max_psi = args.max_psi
DIR = os.path.join(CONSTANTS.GraphDMDDIR, f"rsfMRIfull_window=64_featdim=16_th=0.2_step=4_trial_results_psi={min_psi}_{max_psi}")


all_modes = {}
for i, s in enumerate(filtered_subject_ids):
   modes = np.load(f"{DIR}/b_{i+1}/aligned_modes.npy")
   all_modes[s] = modes

num_modes = 2
N = 5000
for m in range(num_modes):
   netmat = np.zeros((len(subject_ids), N), dtype=float)
   for i, s in enumerate(subject_ids):
      if s in all_modes:
         netmat[i] = all_modes[s][:, m]
   np.savetxt(
      os.path.join(HOME,
                   f"netmats/3T_HCP1200_MSMAll_d{50}_ts2/netmats_cluster_mode_psi={min_psi}_{max_psi}_mode={m}.txt"), netmat)
print("Done!")