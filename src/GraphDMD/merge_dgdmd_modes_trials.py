import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

HOME = f"/Users/mturja/Downloads/HCP_PTN1200"
N = 50
subject_ids = list(np.loadtxt(os.path.join(HOME, "subjectIDs.txt"), dtype=int))
netmat3 = np.zeros((4, len(subject_ids), N * N), dtype=float)
netmat1 = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmats1.txt"))
MIN_PSI = 0
MAX_PSI = 0.01

for TRIAL in range(4):
    corr_list = []
    count = 0
    for i, s in enumerate(subject_ids):
        file = os.path.join(HOME, f"dgdmd/rsfMRIdgdmd_16_{MIN_PSI}_{MAX_PSI}_{TRIAL}", f"{s}.txt")
        if os.path.exists(file):
            netmat3[TRIAL, i] = np.loadtxt(file).flatten()
            corr = stats.pearsonr(netmat1[i], netmat3[TRIAL, i])[0]
            print(f"Corr: {corr}")
            corr_list.append(abs(corr))
            if abs(corr) < 0.5:
                count += 1

            if corr < 0:
                print(f"Changing sign")
                netmat3[TRIAL, i] = -netmat3[TRIAL, i]

    print(f"bad dmd: {count}")
    plt.hist(corr_list)
    plt.show()
    print(count)

netmat3 = netmat3.mean(axis=0)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.imshow(netmat3.mean(axis=0).reshape(N, N))
plt.subplot(1, 2, 1)
plt.imshow(netmat1.mean(axis=0).reshape(N, N))
plt.tight_layout()
plt.show()
plt.hist(netmat3.flatten(), bins=50)
plt.show()
np.savetxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/dgdmd_netmats/netmats3_{MIN_PSI}_{MAX_PSI}_dgdmd.txt"), netmat3)
