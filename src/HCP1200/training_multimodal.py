import argparse
import math

import numpy as np
import os
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

from src.HCP1200.dataloaders import train_valid_test_folds, prepare_fold_data


class RSFBehaviorPrediction:
    def __init__(self, fold_data, parcels, target, q, modality,  **model_params):
        self.seg_ind = None
        self.fold_data = fold_data
        self.model1 = None
        self.model2 = None
        self.bhv_msr = None
        self.modality = modality
        self.subject_ids = None
        self.parcels = parcels
        self.target = target
        self.q = q
        if len(model_params):
            self.model_params = model_params

    def select_edges(self, x, y, q, mode):
        N_sq = self.parcels * self.parcels
        p_val = np.zeros((N_sq,))
        for i in range(N_sq):
            if i % self.parcels == 0:
                p_val[i] = 1.
                continue
            p_val[i] = stats.pearsonr(x[:, i + N_sq * mode], y)[1]
        sig_ind = np.argpartition(p_val, int(N_sq * q))[:int(N_sq * q)] + N_sq * mode
        return sig_ind

    def fit(self, X_train, seg_ind, y_train):
        X_train_1 = X_train[:, seg_ind[0]]
        X_train_2 = X_train[:, seg_ind[1]]
        alpha = self.model_params["alpha"]
        beta = self.model_params["beta"]
        reg1 = ElasticNet(alpha=alpha)
        reg2 = ElasticNet(alpha=beta)
        reg1.fit(X_train_1, y_train)
        y_pred = reg1.predict(X_train_1)
        residual = y_train - y_pred
        reg2.fit(X_train_2, residual)
        self.model1 = reg1
        self.model2 = reg2
        return reg1, reg2

    def predict(self, X_test, seg_ind, y_test):
        X_test_1 = X_test[:, seg_ind[0]]
        X_test_2 = X_test[:, seg_ind[1]]
        CoD1 = self.model1.score(X_test_1, y_test)
        y_pred_1 = self.model1.predict(X_test_1)
        CoD2 = self.model2.score(X_test_2, y_test - y_pred_1)
        y_pred_final = y_pred_1 + self.model2.predict(X_test_2)
        return y_pred_final, CoD1 + CoD2

    def fit_predict(self, X_train, y_train, X_test, y_test):
        seg_ind_1 = self.select_edges(X_train, y_train, q=self.q, mode=0)
        seg_ind_2 = self.select_edges(X_train, y_train, q=self.q, mode=1)
        self.seg_ind = [seg_ind_1, seg_ind_2]
        self.fit(X_train, self.seg_ind, y_train)
        return self.predict(X_test, self.seg_ind, y_test)

    def cross_validation(self):
        cv_scores = []
        test_preds = []
        test_y = []
        for X_train, y_train, X_val, y_val, X_test, y_test, res_test in self.fold_data:
            _, CoD = self.fit_predict(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val)
            cv_scores.append(CoD)
            pred, _, _ = self.test(X_test, self.seg_ind, y_test)
            res = np.array(res_test)
            test_preds.append(pred + res)
            test_y.append(y_test + res)
        return cv_scores, np.stack(test_preds, axis=0).mean(axis=0), test_y[0]

    def test(self, X, seg_ind, y):
        y_pred, CoD = self.predict(X, seg_ind, y)
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
        for a in np.arange(0.00015, 0.00020, 0.0001): # for gdmd 0
        # for a in np.arange(0.00025, 0.00055, 0.0001):
            for b in np.arange(0.00001, 0.00010, 0.00001):
                for q in [0.7]:
                    model_param = {"name": "elastic", "alpha": a, "beta": b}
                    trainer = RSFBehaviorPrediction(
                        fold_data,
                        parcels,
                        target,
                        q=q,
                        modality=2,
                        deconfound=deconfound,
                        **model_param)
                    scores, test_predictions, test_labels = trainer.cross_validation()
                    if max_score < sum(scores):
                        max_score = sum(scores)
                        m_scores = scores
                        best_fit_params = (a, b, q)
                        y_pred = test_predictions
                        y_true = test_labels
        predictions += list(y_pred)
        ground_truth += list(y_true)
        print(f"max_score: {max_score}, params: {best_fit_params}, scores: {m_scores}")
        # CoD = r2_score(y_true=y_true.reshape(-1, 1), y_pred=y_pred)
        # r = np.corrcoef(y_true, y_pred)[0, 1]
        # print(f"Average CoD: {CoD}, Average r: {r}")
        # res[fold_id, 0] = CoD
        # res[fold_id, 1] = r
        # fold_id += 1

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
