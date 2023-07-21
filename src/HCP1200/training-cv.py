import argparse
import math
from statistics import median

import numpy as np
import os
import pandas as pd
from scipy import stats
from operator import itemgetter

from sklearn.model_selection import KFold, train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
import warnings

from src import CONSTANTS

warnings.filterwarnings("ignore")

from src.HCP1200.dataloaders import train_valid_test_folds, prepare_fold_data


class RSFBehaviorPrediction:
    def __init__(self, fold_data, parcels, target, q,  **model_params):
        self.seg_ind = None
        self.fold_data = fold_data
        self.model = None
        self.bhv_msr = None
        self.subject_ids = None
        self.parcels = parcels
        self.target = target
        self.q = q
        if len(model_params):
            self.model_params = model_params

    def select_edges(self, x, y, q):
        N_sq = x.shape[1]
        p_val = np.zeros((N_sq,))
        for i in range(N_sq):
            if i % self.parcels == 0:
                p_val[i] = 1.
                continue
            p_val[i] = stats.pearsonr(x[:, i], y)[1]
        sig_ind = np.argpartition(p_val, int(N_sq * q))[:int(N_sq * q)]
        return sig_ind

    def fit(self, X_train, y_train):
        reg = ElasticNetCV(cv=10, random_state=42, n_jobs=-1, eps=1e-4, n_alphas=200)
        reg.fit(X_train, y_train)
        self.model = reg
        return reg

    def predict(self, X_test, y_test):
        CoD = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        return y_pred, CoD

    def fit_predict(self, X_train, y_train, X_test, y_test, select_edges=True):
        if select_edges:
            seg_ind = self.select_edges(X_train, y_train, q=self.q)
            self.seg_ind = seg_ind
            self.fit(X_train[:, seg_ind], y_train)
            return self.predict(X_test[:, seg_ind], y_test)
        else:
            self.fit(X_train, y_train)
            return self.predict(X_test, y_test)

    def cross_validation(self, select_edges=True):
        X_train, y_train, X_val, y_val, X_test, y_test, res_test = self.fold_data[0]
        X_train = np.concatenate([X_train, X_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)
        pred, CoD = self.fit_predict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        if select_edges:
            _, CoD_train = self.predict(X_train[:, self.seg_ind], y_train)
        else:
            _, CoD_train = self.predict(X_train, y_train)
        res = np.array(res_test)
        pred = pred + res
        y_test = y_test + res
        return CoD, pred, y_test, CoD_train

    def test(self, X, y):
        y_pred, CoD = self.predict(X, y)
        r = np.corrcoef(y, y_pred)[0, 1]
        return y_pred, CoD, r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PredictSM',
        description='Predict behavioral factors from functional connectivity'
    )
    parser.add_argument('--target', type=str, default="CogTotalComp_AgeAdj")
    parser.add_argument('--corr_type', type=str, default="ikeda_dmd")
    parser.add_argument('--l_freq', type=float, default=0.07)
    parser.add_argument('--r_freq', type=float, default=0.08)
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--q', type=float, default=0.3)
    parser.add_argument("--sel_edge", type=bool, default=False)
    args = parser.parse_args()

    HOME = CONSTANTS.CONSTANTS.HOME
    bhv_msr = pd.read_csv(os.path.join(HOME, f"confounded_behav.csv"))
    parcels = 50
    corr_type = args.corr_type
    target = args.target
    cv_fold = 10
    deconfound = True

    folds = train_valid_test_folds(bhv_msr["Subject"].tolist(), n_fold=cv_fold)
    fold_id = 0
    predictions = []
    ground_truth = []
    q = args.q
    train_score = 0
    for fold in folds:
        y_pred = None
        y_true = None
        fold_data = prepare_fold_data(
            fold, parcels, corr_type, target,
            deconfound, args.l_freq, args.r_freq, args.m)
        trainer = RSFBehaviorPrediction(
            fold_data,
            parcels,
            target,
            q=q,
            deconfound=deconfound)
        scores, test_predictions, test_labels, train_scores = trainer.cross_validation(args.sel_edge)
        predictions += list(test_predictions)
        ground_truth += list(test_labels)
        train_score += train_scores
        print(f"train_score: {train_scores}, test_score: {scores}")

    CoD = r2_score(y_true=np.array(ground_truth), y_pred=np.array(predictions).reshape(-1, 1))
    r = np.corrcoef(np.array(ground_truth), np.array(predictions))[0, 1]
    plt.scatter(y=predictions, x=ground_truth)
    plt.xlabel(f"measured {target}")
    plt.ylabel(f"predicted {target}")
    if deconfound:
        plt.title(f"SM-{target}: (r = {r:0.4f}, CoD = {CoD:0.4f}, Deconfounded)")
    else:
        plt.title(f"SM-{target}: (r = {r:0.4f}, CoD = {CoD:0.4f}, Original)")
    plt.tight_layout()
    plt.savefig(f"res_parcel={parcels}_corr_type={corr_type}_deconfound={deconfound}_target={target}_q={q}_m={args.m}.png")
    np.save(f"y_pred_{corr_type}_{args.l_freq}_{args.r_freq}_{q}.npy", np.array(predictions))
    np.save(f"y_true_{corr_type}_{args.l_freq}_{args.r_freq}.npy", np.array(ground_truth))
    print(f"q = {q}, r = {r}, CoD = {CoD}, CoD_train = {train_score / cv_fold}")
