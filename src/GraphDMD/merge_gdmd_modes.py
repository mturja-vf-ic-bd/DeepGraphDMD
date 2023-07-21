import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

from src.CONSTANTS import CONSTANTS

HOME = CONSTANTS.HOME
N = 50
subject_ids = list(np.loadtxt(os.path.join(HOME, "subjectIDs.txt"), dtype=int))
netmat3 = np.zeros((len(subject_ids), N * N), dtype=float)
netmat1 = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmats1.txt"))
MIN_PSI = 0
MAX_PSI = 0.005

corr_list = []
count = 0
for i, s in enumerate(subject_ids):
    file = os.path.join(HOME, f"rsfMRIfull_window=64_featdim=16_th=0.2_step=4_psi=0.0_0.005/avg_netmats", f"{s}.txt")
    if os.path.exists(file):
        netmat3[i] = np.loadtxt(file).flatten()[0:2500]
        # factor = (netmat1[i] / netmat3[i]).sum()
        # print(f"{factor}")
        # netmat3[i] = netmat3[i] * abs(factor)
        corr = stats.pearsonr(netmat1[i], netmat3[i])[0]
        print(f"Corr: {corr}")
        corr_list.append(abs(corr))
        # if abs(corr) < 0.5:
        #     count += 1

        if corr < 0:
            print(f"Changing sign")
            netmat3[i] = -netmat3[i]

print(f"bad dmd: {count}")
plt.hist(corr_list)
print(np.mean(corr_list), np.std(corr_list))
plt.show()
print(count)
plt.imshow(netmat3.mean(axis=0).reshape(N, N))
plt.show()
plt.hist(netmat3.flatten(), bins=50)
plt.show()
np.savetxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmats3_{MIN_PSI}_{MAX_PSI}_aligned.txt"), netmat3)
