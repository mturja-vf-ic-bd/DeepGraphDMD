import scipy.io
import torch
import numpy as np

from src.GraphDMD.utils import process_matlab_cells_single

Phi, Psi, Lambda, Omega, B0, ttX = process_matlab_cells_single(
    "/Users/mturja/Downloads/rsfMRI_global_16_0.15",
    id="b_1", loadphi=True)
print(len(Phi))
Lambda_pt = [np.power(Lambda, t) for t in range(0, 100)]
Lambda_pt = np.stack(Lambda_pt, axis=0)

Phi = torch.from_numpy(Phi).permute(2, 0, 1)
Psi = torch.from_numpy(Psi).unsqueeze(-1)
Lambda_pt = torch.from_numpy(Lambda_pt).unsqueeze(-1)
B0 = torch.from_numpy(B0).unsqueeze(-1).unsqueeze(0)
print(Phi.shape)
r = 50
Lambda_pt = torch.mul(Lambda_pt, B0)
res = torch.mul(Phi[0:r], Lambda_pt[:, 0:r]).sum(dim=1)


from matplotlib import pyplot as plt
plt.figure(figsize=(40, 4))
for i in range(0, 20):
    plt.subplot(2, 20, i+1)
    plt.imshow(torch.real(res[i]))
    plt.subplot(2, 20, i + 21)
    plt.imshow(ttX[:, :, i])
plt.tight_layout()
plt.show()
