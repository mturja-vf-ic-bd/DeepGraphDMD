# DeepGraphDMD
This repository contains code for the method ***"DeepGraphDMD: Interpretable Spatio-Temporal Decomposition of Non-linear Functional Brain Network Dynamics"*** which is accepted in MICCAI 2023.
![Illustration](fig_illustration.png)

## Dataset
Download HCP resting-state data from [HCP1200 Parcellation+Timeseries+Netmats (PTN)](https://db.humanconnectome.org/data/projects/HCP_1200). The time-series data used in this project resides in the folder ```HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d{parcel}_ts2``` (for the experiments in the paper, ```parcel=50``` but other values can also be used).

## Training
To train the model, ```SparseEdgeKoopman/trainer.py``` file is used in the following manner:
```
python3 -m src.SparseEdgeKoopman.trainer --weight 1 10 0.5 --hidden_dim 64 -g 1 -m 500 -d 0.1 --lr 1e-4 --mode train --write_dir SparseEdgeKoopman/megatrawl/win=16_lkis=64 --window 16 --lkis_window 64 --batch_size 16 --stride 4 --latent_dim 32 &
```

## Prediction (Generate latent network embedding that has linear dynamics)
After the model is trained, generate a latent network sequence for every subject using the following command:
```
python3 -m src.SparseEdgeKoopman.predict --subject_id <subject_id> --window 16 --stride 4 --trial 0
```

## Applying GraphDMD to extract DMD modes and their frequency
Run the `Matlab` script: `src/GraphDMD/batch_wise_DeepgDMD.m` to generate DMD modes `Phi` and their frequency `Psi` for every lkis window.
```
batch_wise_DeepgDMD(<subject_id>)
```

To aggregate `Phi` across all lkis windows and generate a set of average `Phi_avg` for each subject, run the following python script:

```
python3 -m src.GraphDMD.cluster_gdmd_modes --subject_id <subject_id> --trial 0 --min_K 8 --max_K 15 --min_psi 0 --max_psi 0.15
```

Now, the aggregated `Phi_avg` can be used to do downstream tasks such as predicting behavioral measures. However, before that run the following script to align the `Phi_avg` modes across subjects (`Phi_aligned`):

```
python3 -m src.GraphDMD.cluster_group_gdmd --min_psi <lower_freq> --max_psi <upper_freq>
```

## Regression Analysis of Behavioral Measures from HCP
Train an ElasticNet regressor, to regress various behavioral measures of HCP data using the aligned DMD modes `Phi_aligned`:

```python3 -m src.HCP1200.training-cv --target <behavioral_measure_name> --corr_type gDMD_multi --l_freq <lower_freq> --r_freq <upper_freq> --q 0.4' --output=training-cv-gDMD_multi_<behavioral_measure_name>_<lower_freq>-<upper_freq>.txt```

Example:
```python3 -m src.HCP1200.training-cv --target CogTotalComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.4' --output=training-cv-gDMD_multi_CogTotalComp_AgeAdj_0.07-0.08_aligned.txt```



