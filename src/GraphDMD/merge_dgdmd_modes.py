import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

HOME = f"/Users/mturja/Downloads/HCP_PTN1200"
N = 50
subject_ids = list(np.loadtxt(os.path.join(HOME, "subjectIDs.txt"), dtype=int))
netmat3 = np.zeros((len(subject_ids), N * N), dtype=float)
netmat4 = np.zeros((len(subject_ids), N * N), dtype=float)
netmat1 = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmats1.txt"))
MIN_PSI = 0
MAX_PSI = 0.01

corr_list = []
count = 0
for i, s in enumerate(subject_ids):
    file = os.path.join(HOME, f"dgDMD_{MIN_PSI}_{MAX_PSI}", f"{s}.txt")
    file2 = os.path.join(HOME, f"dgDMD_{0.075}_{0.085}", f"{s}.txt")
    if os.path.exists(file):
        netmat3[i] = np.loadtxt(file).flatten()
        netmat4[i] = np.loadtxt(file2).flatten()
        # factor = (netmat1[i] / netmat3[i]).sum()
        # print(f"{factor}")
        # netmat3[i] = netmat3[i] * abs(factor)
        corr = stats.pearsonr(netmat4[i], netmat3[i])[0]
        print(f"Corr: {corr}")
        corr_list.append(abs(corr))
        if abs(corr) < 0.5:
            count += 1

        if corr < 0:
            print(f"Changing sign")
            netmat3[i] = -netmat3[i]

        plt.figure(figsize=(15, 5))
        ax = plt.subplot(1, 3, 1)
        ax.imshow(netmat1[i].reshape(50, 50))
        ax.set_title("Pearson")
        ax = plt.subplot(1, 3, 3)
        ax.imshow(netmat4[i].reshape(50, 50))
        ax.set_title("freq=0.075-0.085")
        ax = plt.subplot(1, 3, 2)
        ax.imshow(netmat3[i].reshape(50, 50))
        ax.set_title("freq=0")
        plt.tight_layout()
        plt.show()

print(f"bad dmd: {count}")
plt.hist(corr_list)
plt.show()
print(count)
plt.imshow(netmat3.mean(axis=0).reshape(N, N))
plt.show()
plt.hist(netmat3.flatten(), bins=50)
plt.show()
np.savetxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmats3_{MIN_PSI}_{MAX_PSI}_dgdmd.txt"), netmat3)
