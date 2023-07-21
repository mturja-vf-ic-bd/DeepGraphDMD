#!/bin/bash

mkdir -p "movement/REST1_RL"
mkdir -p "movement/REST1_LR"
mkdir -p "movement/REST2_RL"
mkdir -p "movement/REST2_LR"

for sub in `cat subjectIDs.txt`; do
    aws s3 cp s3://hcp-openaccess/HCP_1200/${sub}/MNINonLinear/Results/rfMRI_REST1_RL/Movement_RelativeRMS_mean.txt movement/REST1_RL/${sub}.txt;
    aws s3 cp s3://hcp-openaccess/HCP_1200/${sub}/MNINonLinear/Results/rfMRI_REST1_LR/Movement_RelativeRMS_mean.txt movement/REST1_LR/${sub}.txt;
    aws s3 cp s3://hcp-openaccess/HCP_1200/${sub}/MNINonLinear/Results/rfMRI_REST2_RL/Movement_RelativeRMS_mean.txt movement/REST2_RL/${sub}.txt;
    aws s3 cp s3://hcp-openaccess/HCP_1200/${sub}/MNINonLinear/Results/rfMRI_REST2_LR/Movement_RelativeRMS_mean.txt movement/REST2_LR/${sub}.txt;
done
