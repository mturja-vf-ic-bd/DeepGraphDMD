#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:39:12 2020

@author: shigeyukiikeda
"""

import pandas
import statsmodels.api as sm
import os

from src.CONSTANTS import CONSTANTS

HOME = CONSTANTS.HOME


# screening subjects based on their behavior and head motion during fMRI scanning
def screening_subject(hcp_restricted_behav_data_path, rest_name, columns):
    # load HCP behavioral data
    behav_data = pandas.read_csv(filepath_or_buffer=hcp_restricted_behav_data_path)

    # screening using notna against all
    behav_data = behav_data[columns]
    behav_data = behav_data.dropna(axis='index',
                                   how='any')

    # screening using 3T_RS-fMRI_PctCompl
    behav_data = behav_data[behav_data['3T_RS-fMRI_PctCompl'] == 100]

    # screening using MMSE score
    behav_data = behav_data[behav_data[
                                'MMSE_Score'] >= 27]  # see "Detecting Dementia With the Mini-Mental State Examination in Highly Educated Individuals"

    # screening subjects whose race are unknown
    # behav_data = behav_data[behav_data['Race'] != 'Unknown or Not Reported']

    # transform subject id to str and set it to index
    behav_data = behav_data.astype(dtype={'Subject': str}).set_index(keys='Subject')

    # screening using movement parameter
    head_motion = pandas.DataFrame(index=behav_data.index,
                                   columns=rest_name,
                                   dtype='float64')
    for sbj in behav_data.index:
        for rest in rest_name:
            with open(os.path.join(HOME, "movement", rest, f"{sbj}.txt"),
                      'rt') as f:
                meanmotion = f.read().replace('\n', '')
            head_motion.loc[sbj, rest] = float(meanmotion)
    behav_data = behav_data.assign(mean_head_motion=head_motion.mean(axis='columns'))
    behav_data = behav_data[
        (head_motion <= 0.15).all(axis='columns')]  # no previous studies, see 10.1038/nn.4135 and 10.1002/hbm.23890

    return behav_data


# regress out from conc_behav_data
def regressout(behav_data,
               target_behav_name,
               nuisance_cov_name):
    # change nominal scale to [0 or 1]
    X = pandas.get_dummies(data=behav_data[nuisance_cov_name],
                           drop_first=True)
    X["Age2"] = X["Age_in_Yrs"] ** 2
    X["Gender-Age"] = X["Gender_M"] * X["Age_in_Yrs"]
    X["Gender-Age2"] = X["Gender_M"] * X["Age2"]

    # add intercept
    X = sm.add_constant(data=X)

    # regress out
    ro_behav_data = pandas.DataFrame(index=behav_data.index,
                                     columns=target_behav_name,
                                     dtype='float64')
    regressor = {}
    for behav in target_behav_name:
        model = sm.OLS(endog=behav_data[behav],
                         exog=X,
                         missing='raise')
        results = model.fit()
        ro_behav_data[behav] = results.resid
        regressor[behav] = results

    return ro_behav_data, regressor


def regressout_predict(behav_data,
                       target_behav_name,
                       nuisance_cov_name,
                       model):
    # change nominal scale to [0 or 1]
    X = pandas.get_dummies(data=behav_data[nuisance_cov_name],
                           drop_first=True)
    X["Age2"] = X["Age_in_Yrs"] ** 2
    X["Gender-Age"] = X["Gender_M"] * X["Age_in_Yrs"]
    X["Gender-Age2"] = X["Gender_M"] * X["Age2"]

    # add intercept
    X = sm.add_constant(data=X)

    # regress out
    ro_behav_data = pandas.DataFrame(index=behav_data.index,
                                     columns=target_behav_name,
                                     dtype='float64')
    for behav in target_behav_name:
        ro_behav_data[behav] = behav_data[behav] - X.dot(model.params)
    return ro_behav_data, X.dot(model.params)


# execute main process
def main():
    ############################################################
    # parameter setup
    hcp_restricted_behav_data_path = os.path.join(HOME, f"Behavioural_HCP_S1200.csv")

    rest_name = ['REST1_LR',
                 'REST1_RL',
                 'REST2_LR',
                 'REST2_RL']

    # use 6 subject data

    # Use CogFluidComp_Unadj and 58 behavioral data
    target_behav_name = ['CogFluidComp_Unadj',
                         'CogFluidComp_AgeAdj',
                         'CogTotalComp_Unadj',
                         'CogTotalComp_AgeAdj',
                         'PicSeq_Unadj',
                         'PicSeq_AgeAdj',
                         'CardSort_Unadj',
                         'CardSort_AgeAdj',
                         'Flanker_Unadj',
                         'Flanker_AgeAdj',
                         'PMAT24_A_CR',
                         'ReadEng_Unadj',
                         'ReadEng_AgeAdj',
                         'PicVocab_Unadj',
                         'ProcSpeed_Unadj',
                         'ProcSpeed_AgeAdj',
                         'DDisc_AUC_40K',
                         'VSPLOT_TC',
                         'SCPT_SEN',
                         'SCPT_SPEC',
                         'IWRD_TOT',
                         'ListSort_Unadj',
                         'MMSE_Score',
                         'PSQI_Score',
                         'Endurance_Unadj',
                         'GaitSpeed_Comp',
                         'Dexterity_Unadj',
                         'Strength_Unadj',
                         'Odor_Unadj',
                         'PainInterf_Tscore',
                         'Taste_Unadj',
                         'Mars_Final',
                         'Emotion_Task_Face_Acc',
                         'Language_Task_Math_Avg_Difficulty_Level',
                         'Language_Task_Story_Avg_Difficulty_Level',
                         'Relational_Task_Acc',
                         'Social_Task_Perc_Random',
                         'Social_Task_Perc_TOM',
                         'WM_Task_Acc',
                         'NEOFAC_A',
                         'NEOFAC_O',
                         'NEOFAC_C',
                         'NEOFAC_N',
                         'NEOFAC_E',
                         'ER40_CR',
                         'ER40ANG',
                         'ER40FEAR',
                         'ER40HAP',
                         'ER40NOE',
                         'ER40SAD',
                         'AngAffect_Unadj',
                         'AngHostil_Unadj',
                         'AngAggr_Unadj',
                         'FearAffect_Unadj',
                         'FearSomat_Unadj',
                         'Sadness_Unadj',
                         'LifeSatisf_Unadj',
                         'MeanPurp_Unadj',
                         'PosAffect_Unadj',
                         'Friendship_Unadj',
                         'Loneliness_Unadj',
                         'PercHostil_Unadj',
                         'PercReject_Unadj',
                         'EmotSupp_Unadj',
                         'InstruSupp_Unadj',
                         'PercStress_Unadj',
                         'SelfEff_Unadj']

    nuisance_cov_name = ['Age_in_Yrs',
                         'Gender',
                         'mean_head_motion']
    ############################################################

    ############################################################
    # main process

    # data screening
    behav_data = screening_subject(hcp_restricted_behav_data_path=hcp_restricted_behav_data_path, rest_name=rest_name, columns=target_behav_name + ["Subject", "Gender", "Age", "3T_RS-fMRI_PctCompl"])
    behav_data["Age_in_Yrs"] = behav_data["Age"].str.split("-", expand=True)[0].astype(int)
    behav_data.to_csv(os.path.join(HOME, "confounded_behav" + '.csv'))
    # regress out nuisance covariates from behavior
    ro_behav_data, _ = regressout(behav_data=behav_data,
                               target_behav_name=target_behav_name,
                               nuisance_cov_name=nuisance_cov_name)

    # save behav_hcp
    ro_behav_data.to_csv(os.path.join(HOME,  "regressed_behav" + '.csv'))
    ############################################################


if __name__ == '__main__':
    main()