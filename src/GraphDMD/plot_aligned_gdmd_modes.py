import os

import numpy as np
from matplotlib import pyplot as plt


DIR = "/Users/mturja/PycharmProjects/KVAE/src/GraphDMD/gdmd_avg_modes"


def convert_flat_mode_to_network(flat_mode):
    N = int(np.sqrt(flat_mode.shape[0] // 2))
    real = flat_mode[0:N*N].reshape(N, N)
    imag = flat_mode[N*N:].reshape(N, N)
    return real, imag


for file in os.listdir(DIR):
    filename = os.path.join(DIR, file, "aligned_modes.npy")
    modes = np.load(filename)
    fig = plt.figure(figsize=(20, 10 * modes.shape[1]))
    for i in range(modes.shape[1]):
        real, imag = convert_flat_mode_to_network(modes[:, i])
        ax = fig.add_subplot(modes.shape[1], 2, 2*i + 1)
        ax.imshow(real)
        ax.title.set_text("Real")
        ax = fig.add_subplot(modes.shape[1], 2, 2*i + 2)
        ax.imshow(imag)
        ax.title.set_text("Imag")
    plt.tight_layout()
    plt.show()

