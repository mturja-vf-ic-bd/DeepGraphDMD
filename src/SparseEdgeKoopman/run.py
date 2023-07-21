window = 16
EPOCH = 500
lkis_window = [64, 128]
K = [41, 41]
cmd = []
for lkis_win, k in zip(lkis_window, K):
    cmd.append( f"python3 -m src.SparseEdgeKoopman.trainer --weight 1 10 0.5 --hidden_dim 64 -g 1 -m {EPOCH} -d 0.1 --lr 1e-4 --mode train --write_dir SparseEdgeKoopman/win={window}_lkis={lkis_win}_k={k} --window {window} --lkis_window {lkis_win} --k {k} --batch_size 4 --stride 3 --latent_dim 128")

with open("run.sh", "w") as f:
    f.write("#!/bin/bash")
    f.write("\n")

    for c in cmd:
        f.writelines(c)
        f.write(" &")
        f.write("\n")

