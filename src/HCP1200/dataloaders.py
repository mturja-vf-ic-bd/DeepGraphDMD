from operator import itemgetter

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold

from src.HCP1200.load_subject_demographic import regressout, regressout_predict
from src.CONSTANTS import CONSTANTS

HOME = CONSTANTS.HOME
bhv_msr = pd.read_csv(os.path.join(HOME, f"confounded_behav.csv"))


def load_data(ids, parcels, corr_type, target, deconfound, mode="train", reg=None, l_freq=0.07, r_freq=0.08, m=1):
    def load_edge_data():
        subject_ids = list(np.loadtxt(os.path.join(HOME, "subjectIDs.txt"), dtype=int))
        if corr_type == "partial":
            edge_data = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmats2.txt"))
        elif corr_type == "gDMD":
            edge_data = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmats3_{l_freq}_{r_freq}_aligned.txt"))
        elif corr_type == "gDMD_multi":
            edge_data = np.loadtxt(
                os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmat3_0-0.01_{l_freq}-{r_freq}.txt"))
        elif corr_type == "dgDMD":
            # print("Loading deep gDMD modes ... ")
            edge_data = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/dgdmd_netmats/netmats3_{l_freq}_{r_freq}_dgdmd.txt"))
        elif corr_type == "dgDMD_multi":
            edge_data = np.loadtxt(
                os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmat3_0-0.01_{l_freq}-{r_freq}_dgdmd.txt"))
        elif corr_type == "ikeda_dmd":
            edge_data = []
            for s in subject_ids:
                edge_data.append(np.load(os.path.join(HOME, f"ikeda_modes_summary/modes_{s}_32_trial=0.npy")))
            edge_data = np.stack(edge_data, axis=0)
        elif corr_type == "cluster_gDMD":
            edge_data = np.loadtxt(
                os.path.join(HOME,
                             f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmats_cluster_mode_psi={l_freq}_{r_freq}_mode={m}.txt"))
            edge_data = edge_data[:, 0:2500]
            edge_data = edge_data.reshape(edge_data.shape[0], 50, 50)
            idx = np.triu_indices(50, k=1)
            edge_data = edge_data[:, idx[0], idx[1]]
        elif corr_type == "cluster_gDMD_multi":
            edge_data = np.loadtxt(
                os.path.join(HOME,
                             f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmats_cluster_mode_psi={l_freq}_{r_freq}_mode={m}.txt"))
            edge_data = edge_data[:, 0:2500]
            edge_data = edge_data.reshape(edge_data.shape[0], 50, 50)
            idx = np.triu_indices(50, k=1)
            edge_data = edge_data[:, idx[0], idx[1]]
            edge_data_zero = np.loadtxt(
                os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmats3_0_0.005.txt"))
            edge_data_zero = edge_data_zero.reshape(edge_data.shape[0], 50, 50)
            idx = np.triu_indices(50, k=1)
            edge_data_zero = edge_data_zero[:, idx[0], idx[1]]
            edge_data = np.concatenate([edge_data, edge_data_zero], axis=1)
        elif corr_type == "ica_1":
            edge_data = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmats_ica_1.txt"))
        elif corr_type == "pca_1":
            edge_data = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmats_pca_1.txt"))
        elif corr_type == "exact_dmd":
            edge_data = np.loadtxt(os.path.join(HOME, "ikeda_modes", "netmats_dmd.txt"))
        else:
            edge_data = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{parcels}_ts2/netmats1.txt"))
        idx = np.array([subject_ids.index(s) for s in ids])
        edge_data = edge_data[idx]
        return edge_data

    if l_freq == 0.0:
        l_freq = int(l_freq)
    edge_data = load_edge_data()

    def get_behavioral_data(mode="train", reg=None):
        bhv_msr = pd.read_csv(os.path.join(HOME, f"confounded_behav.csv"))
        bhv_msr = bhv_msr[bhv_msr["Subject"].isin(ids)]
        residuals = None
        if deconfound:
            bhv_msr.set_index("Subject", inplace=True)
            regressor = None
            nuisance_cov_name = ['Age_in_Yrs',
                                 'Gender',
                                 'mean_head_motion']
            if mode == "train":
                ro_behav_data, regressor = regressout(behav_data=bhv_msr,
                                                      target_behav_name=[target],
                                                      nuisance_cov_name=nuisance_cov_name)
            else:
                ro_behav_data, residuals = regressout_predict(behav_data=bhv_msr,
                                                              target_behav_name=[target],
                                                              nuisance_cov_name=nuisance_cov_name,
                                                              model=reg)
            ro_behav_data.reset_index(inplace=True)
            return ro_behav_data, regressor, residuals
        else:
            return bhv_msr

    regressor = None
    residual = None
    if deconfound:
        bhv_msr, regressor, residual = get_behavioral_data(mode, reg)
    else:
        bhv_msr = get_behavioral_data(mode, reg)
    target_sm = [bhv_msr[bhv_msr["Subject"] == s][target].iloc[0] for s in ids]
    if residual is not None:
        residual = residual.to_frame().reset_index(inplace=False)
        residual = [residual[residual["Subject"] == s][0].iloc[0] for s in ids]
    return edge_data, target_sm, regressor, residual


def train_test_kFold(ids, n_fold):
    kFold = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in kFold.split(ids):
        train_ids = itemgetter(*train_idx)(ids)
        test_ids = itemgetter(*test_idx)(ids)
        folds.append((train_ids, test_ids))
    return folds


def train_valid_test_folds(ids, n_fold):
    folds = []
    train_test_folds = train_test_kFold(ids, n_fold)
    for train_ids, test_ids in train_test_folds:
        cv_folds = []
        train_valid_folds = train_test_kFold(train_ids, n_fold)
        for train, valid in train_valid_folds:
            cv_folds.append((list(train), list(valid)))
        folds.append((cv_folds, list(test_ids)))
    return folds


def prepare_data(train, val, test, parcels, corr_type, target, deconfound, l_freq, r_freq, m):
    trainX, trainY, reg, _ = load_data(train, parcels=parcels, corr_type=corr_type, target=target,
                                         mode="train", reg=None, deconfound=deconfound, l_freq=l_freq, r_freq=r_freq, m=m)
    valX, valY, _, _ = load_data(val, parcels=parcels, corr_type=corr_type, target=target,
                                         deconfound=deconfound, mode="val", reg=reg[target], l_freq=l_freq, r_freq=r_freq, m=m)
    testX, testY, _, testRes = load_data(test, parcels=parcels, corr_type=corr_type, target=target,
                              deconfound=deconfound, mode="test", reg=reg[target], l_freq=l_freq, r_freq=r_freq, m=m)
    return trainX, trainY, valX, valY, testX, testY, testRes


def prepare_fold_data(fold, parcels, corr_type, target, deconfound, l_freq=None, r_freq=None, m=1):
    cv_folds = fold[0]
    test_fold = fold[1]
    fold_data = []
    for cv in cv_folds:
        fold_data.append(prepare_data(cv[0], cv[1], test_fold, parcels, corr_type, target, deconfound, l_freq, r_freq, m))
    return fold_data


def load_node_timeseries():
    import scipy.io
    node_ts = np.zeros((len(bhv_msr["Subject"].tolist()), 4800, 50))
    for i, s in enumerate(bhv_msr["Subject"].tolist()):
        node_ts[i] = np.loadtxt(os.path.join(HOME, "node_timeseries", "3T_HCP1200_MSMAll_d50_ts2", f"{s}.txt"))
    scipy.io.savemat(f"rsfMRI.mat", {"X": node_ts})


load_node_timeseries()
# folds = train_valid_test_folds(bhv_msr["Subject"].tolist(), n_fold=10)
# print("h")
# fold_data = []
# for fold in folds:
#     fold_data.append(prepare_fold_data(fold, 50, 'partial', 'CogTotalComp_Unadj', True))
# print(len(fold_data))

