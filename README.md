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
