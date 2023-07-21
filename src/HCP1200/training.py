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
        if self.model_params["name"] == "linear-reg":
            reg = LinearRegression()
        elif self.model_params["name"] == "elastic":
            alpha = self.model_params["alpha"]
            reg = ElasticNet(alpha=alpha)
        elif self.model_params["name"] == "elastic-cv":
            reg = ElasticNetCV(cv=10, random_state=42)
        elif self.model_params["name"] == "ridge":
            alpha = self.model_params["alpha"]
            reg = Ridge(alpha=alpha)
        elif self.model_params["name"] == "kernel_ridge":
            alpha = self.model_params["alpha"]
            reg = KernelRidge(alpha=alpha)
        else:
            return None
        reg.fit(X_train, y_train)
        self.model = reg
        return reg

    def predict(self, X_test, y_test):
        CoD = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        return y_pred, CoD

    def fit_predict(self, X_train, y_train, X_test, y_test):
        seg_ind = self.select_edges(X_train, y_train, q=self.q)
        self.seg_ind = seg_ind
        self.fit(X_train[:, seg_ind], y_train)
        return self.predict(X_test[:, seg_ind], y_test)

    def cross_validation(self):
        cv_scores = []
        test_preds = []
        test_y = []
        for X_train, y_train, X_val, y_val, X_test, y_test, res_test in self.fold_data:
            _, CoD = self.fit_predict(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val)
            cv_scores.append(CoD)
            pred, _, _ = self.test(X_test[:, self.seg_ind], y_test)
            res = np.array(res_test)
            test_preds.append(pred + res)
            test_y.append(y_test + res)
        return cv_scores, np.stack(test_preds, axis=0).mean(axis=0), test_y[0]

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
    args = parser.parse_args()
    print(args)

    HOME = f"/Users/mturja/Downloads/HCP_PTN1200"
    bhv_msr = pd.read_csv(os.path.join(HOME, f"confounded_behav.csv"))
    parcels = 50
    corr_type = "gDMD"
    target = args.target
    cv_fold = 10
    deconfound = True

    folds = train_valid_test_folds(bhv_msr["Subject"].tolist(), n_fold=cv_fold)
    fold_id = 0
    predictions = []
    ground_truth = []
    for fold in folds:
        max_score = -math.inf
        m_scores = None
        best_fit_params = None
        y_pred = None
        y_true = None
        fold_data = prepare_fold_data(fold, parcels, corr_type, target, deconfound)
        a = None
        for a in np.arange(0.00015, 0.0002, 0.0001):
            for q in [0.7]:
                model_param = {"name": "elastic-cv", "alpha": a}
                trainer = RSFBehaviorPrediction(
                    fold_data,
                    parcels,
                    target,
                    q=q,
                    deconfound=deconfound,
                    **model_param)
                scores, test_predictions, test_labels = trainer.cross_validation()
                if max_score < sum(scores):
                    max_score = sum(scores)
                    m_scores = scores
                    best_fit_params = (a, q)
                    y_pred = test_predictions
                    y_true = test_labels
        predictions += list(y_pred)
        ground_truth += list(y_true)
        print(f"max_score: {max_score}, params: {best_fit_params}, scores: {m_scores}")

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
    plt.savefig(f"res_parcel={parcels}_corr_type={corr_type}_deconfound={deconfound}_target={target}.png")
    np.save("y_pred.npy", np.array(predictions))
    np.save("y_true.npy", np.array(ground_truth))
    print(f"r = {r}, CoD = {CoD}")
