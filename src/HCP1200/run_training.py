import numpy as np

SM_LIST = ["CogTotalComp_AgeAdj", "CogFluidComp_AgeAdj", "PMAT24_A_CR", "ReadEng_AgeAdj", "DDisc_AUC_40K", "CardSort_AgeAdj", "Flanker_AgeAdj"]
# SM_LIST = ["CogFluidComp_AgeAdj", "PMAT24_A_CR"]
CORR_TYPE = "gDMD_multi"
l_freq = 0.07
r_freq = 0.08
m = 0

Q = np.arange(40, 100, 10, dtype=int) * 0.01
with open("run_training.sh", "w") as f:
    f.writelines("#!/bin/bash\n\n")
    for SM in SM_LIST:
        for q in Q:
            if CORR_TYPE == "gDMD_multi" or CORR_TYPE == "dgDMD_multi":
                f.writelines(f"sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 "
                             f"--wrap='python3 -m src.HCP1200.training-cv --target {SM} --corr_type {CORR_TYPE} "
                             f"--l_freq {l_freq} --r_freq {r_freq} --q {q}' "
                             f"--output=training-cv-{CORR_TYPE}_{SM}_{l_freq}-{r_freq}_aligned.txt\n")
            else:
                f.writelines(f"sbatch -p general -N 1 -n 32 --mem=16g -t 12:00:00 "
                             f"--wrap='python3 -m src.HCP1200.training-cv --target {SM} --corr_type {CORR_TYPE} "
                             f"--l_freq {l_freq} --r_freq {r_freq} --q {q} --m {m}' "
                             f"--output=training-cv-{CORR_TYPE}_{SM}_{l_freq}-{r_freq}_{m}.txt\n")