import torch
import torch.nn as nn

from src.SparseEdgeKoopman.utils import compute_window_lkis_loss, corr_coeff, compute_adjacency_pred_loss
from src.CONSTANTS import CONSTANTS

import os


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()


class SparseEdgeKoopman(nn.Module):
    def __init__(
            self,
            feat_dim,
            hidden_dim,
            latent_dyn_dim,
            num_nodes,
            dropout,
            lkis_loss_win=10,
            k=16,
            sp_rat=0.1,
            stride=5, topK=False):
        super(SparseEdgeKoopman, self).__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lkis_loss_win = lkis_loss_win
        self.k = k
        self.topK = topK
        self.num_nodes = num_nodes
        self.sp_rat = sp_rat
        self.nem = torch.load(open(os.path.join(CONSTANTS.CODEDIR, f"src/SparseEdgeKoopman/nem_{num_nodes}.pt"), "rb"))
        # self.nem = torch.load(open(f"nem_{num_nodes}.pt", "rb"))
        self.nem = self.nem[:, :, 0] * num_nodes + self.nem[:, :, 1]
        self.encoder = nn.Sequential(
            # self.drop,
            nn.Linear(feat_dim, hidden_dim),
            # nn.BatchNorm2d(149),
            nn.ReLU(),
            # self.drop,
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            # self.drop,
            nn.Linear(hidden_dim, latent_dyn_dim)
        )
        self.decoder = nn.Sequential(
            # self.drop,
            nn.Linear(latent_dyn_dim, hidden_dim),
            # nn.BatchNorm2d(149),
            nn.ReLU(),
            # self.drop,
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # self.drop,
            nn.Linear(hidden_dim, feat_dim)
        )
        self.stride = stride
        self.apply(_init_weights)

    def forward(self, X):
        A = corr_coeff(X)
        Z = self.encoder(X)
        G = corr_coeff(Z)
        G_u_flat = G.flatten(start_dim=2)
        G_u = G_u_flat[:, :, self.nem]

        if self.topK:
            topK_idx = torch.topk(A.flatten(start_dim=2)[:, :, self.nem], dim=-1, k=self.k)[1]
            G_u_topK = torch.gather(G_u, dim=-1, index=topK_idx)
            loss_rss_all = compute_window_lkis_loss(
                G_u_topK, w=self.lkis_loss_win, N=X.shape[2],
                stride=1)
        else:
            loss_rss_all = compute_window_lkis_loss(
                G_u, w=self.lkis_loss_win, N=X.shape[2],
                stride=1)

        # Find recon loss
        X_recon = self.decoder(Z)
        recon_loss = nn.MSELoss()(X, X_recon)
        adj_loss = compute_adjacency_pred_loss(G, A, k=int(self.num_nodes * self.num_nodes * self.sp_rat))
        output_dict = {"input": X, "recon": X_recon, "z_gauss": Z,
                       "recon_loss": recon_loss, "adj_loss": adj_loss,
                       "loss_rss": loss_rss_all}
        return output_dict


import unittest


class testSparseEdgeKoopman(unittest.TestCase):
    def testOutputShape(self):
        x = torch.randn((2, 100, 22, 10))
        model = SparseEdgeKoopman(
            feat_dim=10,
            hidden_dim=32,
            latent_dyn_dim=64,
            num_nodes=22,
            stride=3, dropout=0.5, lkis_loss_win=64)
        out = model(x)
        self.assertEqual(out["recon"].shape, x.shape)
        self.assertEqual(out["z_gauss"].shape, (2, 100, 22, 64))