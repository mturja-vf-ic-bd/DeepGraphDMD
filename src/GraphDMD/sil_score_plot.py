import os.path
import numpy as np

from src.CONSTANTS import CONSTANTS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--min_psi", type=float, default=0.09,
                    help="Minimum frequency for which DMD modes will be considered")
parser.add_argument("--max_psi", type=float, default=0.12,
                    help="Maximum frequency for which DMD modes will be considered")
args = parser.parse_args()
DIR_SUFFIX = CONSTANTS.GraphDMDDIR + f"/rsfMRIfull_window=64_featdim=16_th=0.2_step=4_trial_results_psi={args.min_psi}_{args.max_psi}"

scores = []
for b in range(1, 841):
    fname = os.path.join(DIR_SUFFIX, f"b_{b}/Sil_score.npy")
    if os.path.exists(fname):
        scores.append(np.load(fname))

scores = np.stack(scores, axis=0)
stats = np.stack([np.min(scores, axis=0),
                  np.percentile(scores, q=25, axis=0),
                  np.median(scores, axis=0),
                  np.percentile(scores, q=75, axis=0),
                  np.max(scores, axis=0)], axis=0)
np.set_printoptions(precision = 5, suppress = True)
np.savetxt(f"stats_{args.min_psi}_{args.max_psi}.csv", stats[:, 0:5].T, delimiter=",")
print(list(stats[:, 0:5].T))
print(scores.shape)


