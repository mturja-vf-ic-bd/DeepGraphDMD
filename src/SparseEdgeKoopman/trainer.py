import os
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from src.SparseEdgeKoopman.loaders.data_loader import MegaTrawlDataModule
from src.SparseEdgeKoopman.model.SparseEdgeKoopmanModel import SparseEdgeKoopman
from src.utils.folder_manager import create_version_dir
from src.CONSTANTS import CONSTANTS


class trainerSparseEdgeKoopman(pl.LightningModule):
    def __init__(self,
                 hidden_dim,
                 latent_dim,
                 window=16,
                 num_nodes=4,
                 lr=1e-4,
                 dropout=0.5,
                 loss_weights=None,
                 stride=5,
                 lkis_window=10,
                 sp_rat=0.1,
                 k=16, topK=False):
        super(trainerSparseEdgeKoopman, self).__init__()

        self.model = SparseEdgeKoopman(
            feat_dim=window,
            hidden_dim=hidden_dim,
            k=k,
            latent_dyn_dim=latent_dim,
            num_nodes=num_nodes,
            dropout=dropout,
            lkis_loss_win=lkis_window,
            stride=stride,
            sp_rat=sp_rat,
            topK=topK
        )
        self.learning_rate = lr
        self.window = window
        self.stride = stride
        if loss_weights is not None:
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [1, 1, 1]
        self.save_hyperparameters()

    def forward(self, input):
        return self.model(input)

    def add_log(self, losses, category="train"):
        for k, v in losses.items():
            self.log(category + "/" + k, v)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=0.001)

    def get_attn(self):
        return self.model.K

    def _common_step(self, batch, batch_idx, category="train"):
        if len(batch) == 3:
            x, corr, y = batch
        elif len(batch) == 2:
            x, s = batch
        else:
            x = batch[0]
        x = x.unfold(-1, self.window, self.stride).permute(0, 2, 1, 3)
        output_dict = self.model(x.float())
        # output_dict["labels"] = y
        if self.current_epoch > 1:
            loss = self.loss_weights[0] * output_dict["recon_loss"] + \
                self.loss_weights[1] * output_dict["loss_rss"] + \
                   self.loss_weights[2] * output_dict["adj_loss"]
        else:
            loss = self.loss_weights[0] * output_dict["recon_loss"] + \
                   self.loss_weights[1] * output_dict["loss_rss"]
        losses = {}
        for k, v in output_dict.items():
            if "loss" in k:
                losses[k] = v
        losses["total_loss"] = loss
        self.add_log(losses, category)
        return loss

    def predict_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, corr, y = batch
        elif len(batch) == 2:
            x, s = batch
        else:
            x = batch[0]
        x = x.unfold(-1, self.window, self.stride).permute(0, 2, 1, 3)
        print(x.shape)
        output_dict = self.model(x.float())
        latent = output_dict["z_gauss"].detach().cpu().numpy()
        parcel = 50
        np.save(
            os.path.join(CONSTANTS.HOME,
                         "latent", f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                         f"{int(s)}.npy"), latent)
        # output_dict["labels"] = y
        return output_dict


def cli_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("-g", "--gpus", nargs="?", type=int, default=0)
    parser.add_argument("-m", "--max_epochs", nargs="?", type=int, default=100)
    parser.add_argument("-d", "--dropout", nargs="?",
                        type=float, default=0.01)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")
    parser.add_argument("--from_ckpt", dest='from_ckpt', default=False, action='store_true')
    parser.add_argument("--exp_name", type=str, default="fmri",
                        help="Name you experiment")
    parser.add_argument("--write_dir", type=str, default="GraphEDM/fMRI")
    parser.add_argument("--weight", type=float, nargs="+")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lkis_window", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="hcp")
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--sp_rat", type=float, default=0.1)
    parser.add_argument("--topK", dest='topK', default=False, action='store_true')

    arguments = parser.parse_args()
    print("Hyper-parameters:")
    for k, v in vars(arguments).items():
        print("{} -> {}".format(k, v))

    _HOME = os.path.expanduser('~')
    data_loader = MegaTrawlDataModule(
        batch_size=arguments.batch_size,
        parcel=arguments.num_nodes
    )

    if arguments.mode == "train":
        # Create write dir
        write_dir = create_version_dir(
            os.path.join(arguments.write_dir, arguments.exp_name),
            prefix="run")
        arguments.write_dir = write_dir
        ckpt = ModelCheckpoint(dirpath=os.path.join(write_dir, "checkpoints"),
                               monitor="val/loss_rss",
                               every_n_epochs=10,
                               save_top_k=1,
                               save_last=True,
                               auto_insert_metric_name=False,
                               filename='epoch-{epoch:02d}-recon_loss-{val/recon_loss:0.3f}-lkis_loss-{'
                                        'val/loss_rss:0.3f}-adj_loss-{val/adj_loss:0.3f}')
        tb_logger = pl_loggers.TensorBoardLogger(write_dir, name="tb_logs")
        trainer = pl.Trainer(gpus=arguments.gpus,
                             max_epochs=arguments.max_epochs,
                             logger=tb_logger,
                             log_every_n_steps=1,
                             callbacks=[ckpt])
        if not arguments.from_ckpt:
            model = trainerSparseEdgeKoopman(
                hidden_dim=arguments.hidden_dim,
                latent_dim=arguments.latent_dim,
                loss_weights=arguments.weight,
                window=arguments.window,
                num_nodes=arguments.num_nodes,
                lr=arguments.lr,
                dropout=arguments.dropout,
                lkis_window=arguments.lkis_window,
                stride=arguments.stride,
                sp_rat=arguments.sp_rat,
                topK=arguments.topK
            )
        else:
            model = trainerSparseEdgeKoopman.load_from_checkpoint(arguments.ckpt)
        trainer.fit(
            model,
            train_dataloaders=data_loader.train_dataloader(),
            val_dataloaders=data_loader.val_dataloader()
        )
        best_model = trainerSparseEdgeKoopman.load_from_checkpoint(ckpt.best_model_path)
        return predict(data_loader.val_dataloader(), best_model), arguments, model
    else:
        data_loader = data_loader.val_dataloader()
        model = trainerSparseEdgeKoopman.load_from_checkpoint(arguments.ckpt)
        return predict(data_loader, model), arguments, model


def predict(data_loader, model):
    trainer = pl.Trainer()
    output = trainer.predict(model, data_loader)
    return output


if __name__ == '__main__':
    start_time = time.time()
    output, arguments, model = cli_main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")
