import numpy as np

from src.dNFC.compute_dNFC_mode import load_data, get_dNFC, get_similarity_score, \
    compute_component_based_score_across_trials
import argparse
from src.SparseEdgeKoopman.utils import corr_coeff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str, default="100307")
    parser.add_argument("--parcel", type=int, default=50)
    parser.add_argument("--modality", type=str, default="dNFC")
    parser.add_argument("--min_psi", type=float, default=None)
    parser.add_argument("--max_psi", type=float, default=None)
    parser.add_argument("--n_cluster", type=int, default=4)
    parser.add_argument("--sld_win", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)

    args = parser.parse_args()

    prsn = []
    if args.modality == "dNFC":
        score = get_similarity_score(
            args.subject_id, args.parcel,
            args.sld_win, args.stride, args.n_cluster)
        print(f"dNFC Score: {score}")
    elif args.modality == "ICA":
        score = compute_component_based_score_across_trials(
            args.subject_id, args.parcel,
            args.sld_win, args.stride, args.n_cluster, "ica")
        print(f"ICA Score: {score}")
    elif args.modality == "PCA":
        score = compute_component_based_score_across_trials(
            args.subject_id, args.parcel,
            args.sld_win, args.stride, args.n_cluster, "pca")
        print(f"PCA Score: {score}")
    elif args.modality == "sFC":
        for trial in range(1, 5):
            bold = load_data(args.subject_id, trial, args.parcel)
            prsn.append(corr_coeff(bold).numpy().reshape(-1, ))
        prsn = np.stack(prsn, axis=1)
        prsn /= np.linalg.norm(prsn, axis=0, keepdims=True)
        score = np.triu(np.dot(prsn.T, prsn), k=1).sum() / (prsn.shape[1] * (prsn.shape[1] - 1) // 2)
        print(f"sFC Score: {score}")

