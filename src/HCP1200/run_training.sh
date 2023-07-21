#!/bin/bash

sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogTotalComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.4' --output=training-cv-gDMD_multi_CogTotalComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogTotalComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.5' --output=training-cv-gDMD_multi_CogTotalComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogTotalComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.6' --output=training-cv-gDMD_multi_CogTotalComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogTotalComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.7000000000000001' --output=training-cv-gDMD_multi_CogTotalComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogTotalComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.8' --output=training-cv-gDMD_multi_CogTotalComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogTotalComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.9' --output=training-cv-gDMD_multi_CogTotalComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogFluidComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.4' --output=training-cv-gDMD_multi_CogFluidComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogFluidComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.5' --output=training-cv-gDMD_multi_CogFluidComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogFluidComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.6' --output=training-cv-gDMD_multi_CogFluidComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogFluidComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.7000000000000001' --output=training-cv-gDMD_multi_CogFluidComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogFluidComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.8' --output=training-cv-gDMD_multi_CogFluidComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CogFluidComp_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.9' --output=training-cv-gDMD_multi_CogFluidComp_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target PMAT24_A_CR --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.4' --output=training-cv-gDMD_multi_PMAT24_A_CR_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target PMAT24_A_CR --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.5' --output=training-cv-gDMD_multi_PMAT24_A_CR_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target PMAT24_A_CR --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.6' --output=training-cv-gDMD_multi_PMAT24_A_CR_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target PMAT24_A_CR --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.7000000000000001' --output=training-cv-gDMD_multi_PMAT24_A_CR_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target PMAT24_A_CR --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.8' --output=training-cv-gDMD_multi_PMAT24_A_CR_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target PMAT24_A_CR --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.9' --output=training-cv-gDMD_multi_PMAT24_A_CR_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target ReadEng_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.4' --output=training-cv-gDMD_multi_ReadEng_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target ReadEng_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.5' --output=training-cv-gDMD_multi_ReadEng_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target ReadEng_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.6' --output=training-cv-gDMD_multi_ReadEng_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target ReadEng_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.7000000000000001' --output=training-cv-gDMD_multi_ReadEng_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target ReadEng_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.8' --output=training-cv-gDMD_multi_ReadEng_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target ReadEng_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.9' --output=training-cv-gDMD_multi_ReadEng_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target DDisc_AUC_40K --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.4' --output=training-cv-gDMD_multi_DDisc_AUC_40K_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target DDisc_AUC_40K --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.5' --output=training-cv-gDMD_multi_DDisc_AUC_40K_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target DDisc_AUC_40K --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.6' --output=training-cv-gDMD_multi_DDisc_AUC_40K_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target DDisc_AUC_40K --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.7000000000000001' --output=training-cv-gDMD_multi_DDisc_AUC_40K_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target DDisc_AUC_40K --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.8' --output=training-cv-gDMD_multi_DDisc_AUC_40K_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target DDisc_AUC_40K --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.9' --output=training-cv-gDMD_multi_DDisc_AUC_40K_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CardSort_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.4' --output=training-cv-gDMD_multi_CardSort_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CardSort_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.5' --output=training-cv-gDMD_multi_CardSort_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CardSort_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.6' --output=training-cv-gDMD_multi_CardSort_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CardSort_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.7000000000000001' --output=training-cv-gDMD_multi_CardSort_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CardSort_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.8' --output=training-cv-gDMD_multi_CardSort_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target CardSort_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.9' --output=training-cv-gDMD_multi_CardSort_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target Flanker_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.4' --output=training-cv-gDMD_multi_Flanker_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target Flanker_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.5' --output=training-cv-gDMD_multi_Flanker_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target Flanker_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.6' --output=training-cv-gDMD_multi_Flanker_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target Flanker_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.7000000000000001' --output=training-cv-gDMD_multi_Flanker_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target Flanker_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.8' --output=training-cv-gDMD_multi_Flanker_AgeAdj_0.07-0.08_aligned.txt
sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 --wrap='python3 -m src.HCP1200.training-cv --target Flanker_AgeAdj --corr_type gDMD_multi --l_freq 0.07 --r_freq 0.08 --q 0.9' --output=training-cv-gDMD_multi_Flanker_AgeAdj_0.07-0.08_aligned.txt
