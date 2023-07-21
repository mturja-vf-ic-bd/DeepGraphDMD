import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import scipy.io
from src.CONSTANTS import CONSTANTS

HOME = CONSTANTS.HOME
N = 50
# MIN_PSI = 0
# MAX_PSI = 0.1
w = 64
# INPUT = f"/pine/scr/m/t/mturja/GraphDMD/rsfMRIfull_{w}_80.15/"
# INPUT = f"/Users/mturja/GraphDMD/rsfMRIfull_64_16_0.15_no_Phi/"
# INPUT = f"/Users/mturja/GraphDMD/rsfMRIdgdmd_64_16_0.15_no_Phi/"
INPUT = f"/Users/mturja/GraphDMD/rsfMRI_global_16_0.15_noPhi"
subject_ids = list(np.loadtxt(os.path.join(HOME, "filteredSubjectIds.txt")))

psi = []
print("Staring psi calculations .. ")
for i, s in enumerate(subject_ids):
    for sw in range(0, 1000):
        filename = os.path.join(INPUT, f"b_{i+1}/sw_{sw + 1}.mat")
        # filename = os.path.join(INPUT, f"{int(s)}/sw_{sw + 1}.mat")
        if not os.path.exists(filename):
            break
        mat = scipy.io.loadmat(filename)
        lamb = mat['Lambda'].flatten()
        idx = np.where(np.abs(lamb) > 0.98)
        psi += mat['Psi'].flatten()[idx].tolist()
        print(i, len(psi))
plt.cla()
plt.hist(psi, bins=100)
plt.savefig(f"freq_hist_gdmd_global.png")

# np.save("psi_population.npy", np.array(psi))
