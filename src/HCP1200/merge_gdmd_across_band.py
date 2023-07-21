import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

HOME = f"/Users/mturja/Downloads/HCP_PTN1200"
N = 50
subject_ids = list(np.loadtxt(os.path.join(HOME, "subjectIDs.txt"), dtype=int))
netmat3 = np.zeros((len(subject_ids), N * N), dtype=float)
freq_ranges = [(0, 0.01), (0.08, 0.09)]
gdmd_dict = []

suffix = ""
for l_freq, r_freq in freq_ranges:
    suffix += f"_{l_freq}-{r_freq}"

for l_freq, r_freq in freq_ranges:
    filename = os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmats3_{l_freq}_{r_freq}.txt")
    gdmd_dict.append(np.loadtxt(filename))

# for i in range(len(gdmd_dict)):
#     factor = np.sqrt((gdmd_dict[i] ** 2).sum(axis=1)[:, np.newaxis])
#     gdmd_dict[i] /= factor

res = np.concatenate(gdmd_dict, axis=1)
np.savetxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmat3{suffix}.txt"), res)
