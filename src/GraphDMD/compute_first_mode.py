import argparse

# Process matlab results
import os.path
from pathlib import Path

import numpy as np

from src.CONSTANTS import CONSTANTS
from src.GraphDMD.cluster_gdmd_modes import read_gdmd_results, modified_k_means_single

parser = argparse.ArgumentParser()
parser.add_argument("--min_psi", type=float, default=0, help="Left boundary of psi segment")
parser.add_argument("--max_psi", type=float, default=0.01, help="Right boundary of psi segment")
parser.add_argument("--subject_id", type=str, help="Subject id according to order in bhv_msr")

args = parser.parse_args()
subject_ids = np.loadtxt(CONSTANTS.CODEDIR + "/src/GraphDMD/filteredSubjectIds.txt")
DIR_SUFFIX = CONSTANTS.GraphDMDDIR + "/rsfMRIfull_window=64_featdim=16_th=0.2_step=4"
OUTPUT_PATH = os.path.join(DIR_SUFFIX + f"_psi={args.min_psi}_{args.max_psi}", "avg_netmats")
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

Phi, Psi, Lmd = [], [], []
for trial in range(1, 5):
    phi, psi, lmd, omega, b0 = read_gdmd_results(
        DIR_SUFFIX + f"_trial={trial}", args.subject_id
    )
    psi = np.concatenate(psi, axis=0)
    lmd = np.concatenate(lmd, axis=0)
    phi = np.concatenate(phi, axis=-1)
    print(phi.shape, psi.shape, lmd.shape)
    sel_idx = np.logical_and(
        np.logical_and(psi[:, 0] < args.max_psi, psi[:, 0] >= args.min_psi),
        np.abs(lmd[:, 0]) > 0.9
    )
    Phi.append(phi[:, :, sel_idx])
    print(Phi[-1].shape)
Phi = np.concatenate(Phi, axis=-1)
Phi = Phi.reshape(-1, Phi.shape[2])
Phi = np.concatenate([np.real(Phi), np.imag(Phi)], axis=0)
Phi = modified_k_means_single(Phi)
print(f"Output shape: {Phi.shape}")
np.savetxt(os.path.join(OUTPUT_PATH, str(int(subject_ids[int(args.subject_id.split("_")[1]) - 1])) + ".txt"), Phi.mean(axis=1))
print(f"Done: {args.subject_id}")
