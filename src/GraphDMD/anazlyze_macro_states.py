import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np

from src.GraphDMD.analyse_cluster import smooth_activation_map
from src.GraphDMD.clutering_algorithms import KMedoidCV_JensenShannon, DBSCAN_JensenShannon, KMeansCV

if __name__ == '__main__':
    DIR = "/Users/mturja/Downloads/rsfMRIfull_window=64_featdim=16_th=0.2_step=4_trial_results_psi=0.06_0.09"
    parser = argparse.ArgumentParser("Program to compute macro-state from activation maps")
    parser.add_argument("--subject_id", type=str, default="b_1")
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()

    activation_map = np.load(
        os.path.join(DIR, args.subject_id, "activation_maps.npy"))

    # Check if it's for particular trial or all (trial == None)
    if args.trial:
        activation_map = activation_map[args.trial - 1]
    else:
        activation_map = activation_map.swapaxes(0, 1).reshape((activation_map.shape[1], -1))
    activation_map = smooth_activation_map(activation_map, w=2)
    activation_map /= (np.linalg.norm(activation_map, axis=0, keepdims=True)+1e-2)
    cluster_range = np.arange(4, 5)
    # cluster_labels, medoids, scores = KMedoidCV_JensenShannon(
    #     activation_map.T, cluster_range=cluster_range)
    cluster_labels, medoids, scores = KMeansCV(
        activation_map.T, cluster_range=cluster_range)
    plt.plot(cluster_range, scores)
    plt.show()

    plt.figure(figsize=(activation_map.shape[1]//2, 5))
    plt.tight_layout()
    plt.subplot(211)
    plt.axis('off')
    plt.imshow(activation_map)
    plt.subplot(212)
    plt.axis('off')
    plt.plot(cluster_labels)
    plt.show()
