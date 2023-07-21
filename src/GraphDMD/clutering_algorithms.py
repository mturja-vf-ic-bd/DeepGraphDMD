from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import jensenshannon
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from coclust.clustering import SphericalKmeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn_extra.cluster import KMedoids
import matplotlib.cm as cm
import os


def agglomorative_clustering(X, dist_th):
    Z = linkage(X.T, 'average', 'cosine')
    plt.figure(figsize=(10, 5))
    dendrogram(Z, p=5, truncate_mode='level')
    plt.show()
    cluster_label = fcluster(Z, dist_th, criterion='distance')
    return cluster_label


def GMM(X, cluster_range):
    def gmm_bic_score(estimator, Z):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        score = -estimator.bic(Z)
        print(f"params: {estimator} -> {score}")
        return score

    param_grid = {
        "n_components": cluster_range,
        "covariance_type": ["full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(X)
    print(grid_search.best_params_)
    return grid_search.best_estimator_.predict(X), grid_search.best_estimator_.means_


def KMedoidCV_JensenShannon(X, cluster_range):
    best_score = -1.1
    best_K = -1
    medoids = None
    best_labels = None
    scores = []
    for n_clusters in cluster_range:
        model = KMedoids(n_clusters=n_clusters,
                         metric=jensenshannon,
                         method="pam",
                         init="k-medoids++")
        model.fit(X)
        cur_score = silhouette_score(X, model.labels_, metric=jensenshannon)
        scores.append(cur_score)
        plot_silhouette(X,
                        model.labels_,
                        n_clusters,
                        metric=jensenshannon)
        print(f"K = {n_clusters}, score = {cur_score}")
        if cur_score > best_score:
            best_K = n_clusters
            best_score = cur_score
            best_labels = model.labels_
            medoids = X[model.medoid_indices_, :].T
    print(f"KMedoid best params: K={best_K}")
    return best_labels, medoids, scores


def DBSCAN_JensenShannon(X):
    model = DBSCAN(eps=0.05, min_samples=20, metric=jensenshannon)
    model.fit(X)
    return model.labels_


def KMeansCV(X, cluster_range):
    best_score = -1.1
    best_K = -1
    centers = None
    best_labels = None
    scores = []
    for n_clusters in cluster_range:
        model = KMeans(n_clusters=n_clusters,
                         init="k-means++")
        model.fit(X)
        cur_score = silhouette_score(X, model.labels_)
        scores.append(cur_score)
        plot_silhouette(X,
                        model.labels_,
                        n_clusters)
        print(f"K = {n_clusters}, score = {cur_score}")
        if cur_score > best_score:
            best_K = n_clusters
            best_score = cur_score
            best_labels = model.labels_
            centers = model.cluster_centers_
    print(f"KMedoid best params: K={best_K}")
    return best_labels, centers, scores


def plot_silhouette(X, cluster_labels, n_clusters, metric="euclidean", save_path=None):
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)
    cur_score = silhouette_score(X, cluster_labels, metric=metric)

    y_lower = 10
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax.set_xlim([-0.1, .5])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=cur_score, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.suptitle(
        f"Silhouette analysis "
        f"for KMeans clustering on "
        f"sample data with n_clusters "
        f"= {n_clusters}, score={cur_score:.3f}",
        fontsize=14,
        fontweight="bold"
    )
    if save_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_path, f"sil-plot_K={n_clusters}.png"))


def SphericalKMeansCV(X, cluster_range, OUTPUT_DIR=None):
    def compute_spherical_means(X, cluster_labels):
        modes = np.zeros((X.shape[1], max(cluster_labels) + 1))
        for c in range(max(cluster_labels) + 1):
            if c in cluster_labels:
                mode = X[cluster_labels == c].mean(axis=0)
                mode /= np.linalg.norm(mode)
                modes[:, c] = mode
        return modes
    best_score = -2
    bestK = 1
    scores = []
    best_labels = None
    best_modes = None
    for n_clusters in cluster_range:
        skm = SphericalKmeans(n_clusters=n_clusters, n_init=10)
        skm.fit(X)
        cluster_labels = np.array(skm.labels_)
        cur_score = silhouette_score(X, cluster_labels, metric="cosine")
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            cur_score
        )
        if cur_score > best_score:
            best_score = cur_score
            bestK = n_clusters
            best_labels = cluster_labels
            best_modes = compute_spherical_means(X, cluster_labels)
        scores.append(cur_score)
        plot_silhouette(X, cluster_labels, n_clusters, metric="cosine", save_path=OUTPUT_DIR)
    plt.figure(figsize=(10, 7))
    plt.plot(cluster_range, np.array(scores))
    plt.title("n_cluster vs sil-score")
    plt.savefig(os.path.join(OUTPUT_DIR, "SilscorevsClustersize.png"))
    # plt.show()
    print(f"Best n_cluster = {bestK}, Best score = {best_score}")
    return best_labels, best_modes, scores, bestK
