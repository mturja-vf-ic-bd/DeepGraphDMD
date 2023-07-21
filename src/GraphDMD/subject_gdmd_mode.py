import argparse

# Process matlab results
import os.path

import numpy as np

from utils import process_matlab_cells_single, get_avg_phi_segment

parser = argparse.ArgumentParser()
parser.add_argument("--min_psi", type=float, default=0, help="Left boundary of psi segment")
parser.add_argument("--max_psi", type=float, default=0.01, help="Right boundary of psi segment")
parser.add_argument("--subject_id", type=int, help="Subject id according to order in bhv_msr")
parser.add_argument("--id_type", type=str, default="gdmd")
parser.add_argument("--input", type=str, help="Input directory")
parser.add_argument("--output", type=str, help="Output directory")

args = parser.parse_args()
subject_ids = np.loadtxt("filteredSubjectIds.txt")

if args.id_type == "gdmd":
    Phi, Psi, Lambda, Omega, B0 = process_matlab_cells_single(args.input, args.subject_id, loadphi=True)
else:
    Phi, Psi, Lambda, Omega, B0 = process_matlab_cells_single(args.input, str(args.subject_id), loadphi=True)
dmd_mode = get_avg_phi_segment(Phi, Psi, Lambda, B0, args.min_psi, args.max_psi)

if args.id_type == "gdmd":
    np.savetxt(os.path.join(args.output, str(int(subject_ids[args.subject_id])) + ".txt"), dmd_mode)
else:
    np.savetxt(os.path.join(args.output, str(args.subject_id) + ".txt"), dmd_mode)
print(f"Done: {args.subject_id}")
