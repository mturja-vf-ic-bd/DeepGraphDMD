from src.GraphDMD.utils import process_matlab_cells_single
from matplotlib import pyplot as plt
import numpy as np

Psi, Lambda, Omega, B0 = process_matlab_cells_single(".", "b_4", loadphi=False)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
circ = plt.Circle((0, 0), radius=1.01, edgecolor='b', facecolor='None')
ax.add_patch(circ)
# circ = plt.Circle((0, 0), radius=0.9, edgecolor='b', facecolor='None')
# ax.add_patch(circ)

Lambda[3, 0] = 0.5 + 0.2j
Lambda[4, 0] = 0.4 + 0.3j
Lambda[5, 0] = 0.5 - 0.2j
Lambda[6, 0] = 0.4 - 0.3j
Lambda[7, 0] = 0.55 + 0.15j
Lambda[8, 0] = 0.55 - 0.15j
Lambda[9, 0] = 0.35 + 0.15j
Lambda[10, 0] = 0.35 - 0.15j
Lambda[11, 0] = 0.51 + 0.17j
Lambda[12, 0] = 0.51 - 0.17j
Lambda[13, 0] = 0.25 + 0.23j
Lambda[14, 0] = 0.25 - 0.23j
Lambda[15, 0] = 0.28 + 0.24j
Lambda[16, 0] = 0.28 - 0.24j
Lambda[17, 0] = 1.03 + 0.2j
Lambda[18, 0] = 1.03 - 0.2j
Lambda[19, 0] = 1.1 + 0.1j
Lambda[20, 0] = 1.1 - 0.1j
Lambda[21, 0] = 0.99 + 0.5j
Lambda[22, 0] = 0.99 - 0.5j

dist = 0
for eig in Lambda[:, 0]:
    print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))
    ax.scatter(eig.real, eig.imag)
    dist += np.abs(eig.imag**2+eig.real**2 - 1)

print(f"Avg dist: {dist/len(Lambda)}")
# Psi, Lambda, Omega, B0 = process_matlab_cells_single(".", "b_1", loadphi=False)
# for eig in Lambda[:, 0]:
#     print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))
#     ax.scatter(eig.real, eig.imag, c='b')

plt.show()