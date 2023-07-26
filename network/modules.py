import os
from typing import Any, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from network.resnet1d.resnet1d import ResNet1D, BasicBlock1D, compute_model_args
import network.dido_preprocessor as dido_preprocessor
from network.resnet1d.loss import loss_mse, loss_frobenius, loss_frobenius_v2, loss_rotation, loss_geodesic, loss_geodesic_into_NLL, loss_mse_into_NLL

class VelocityUnitVectorRegressor(pl.LightningModule):
    def __init__(self, learning_rate, input_dim, output_dim, group_sizes, intermediate_dim):
        super().__init__()
        self.save_hyperparameters()

        self.net = ResNet1D(
                BasicBlock1D, input_dim, output_dim, group_sizes, intermediate_dim
            )
        
        self.learning_rate = learning_rate
        
    def training_step(self, batch, batch_idx):

        # training step
        meas_seg, gt_encoding = batch

        # forward + backward + optimize
        outputs, output_cov = self.net(meas_seg)

        # compute and return loss
        loss = loss_geodesic_into_NLL(pred = outputs, pred_cov=output_cov, targ = gt_encoding, device=self.device, epoch=self.current_epoch)

        self.log("training_loss", loss, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):

        # training step
        meas_seg, gt_encoding = batch

        # forward + backward + optimize
        outputs, output_cov = self.net(meas_seg)

        # compute and return loss
        loss = loss_geodesic_into_NLL(pred = outputs, pred_cov=output_cov, targ = gt_encoding, device=self.device, epoch=self.current_epoch)

        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):

        # training step
        meas_seg, gt_encoding = batch

        # forward + backward + optimize
        outputs, output_cov = self.net(meas_seg)

        # compute and return loss
        loss = loss_geodesic_into_NLL(pred = outputs, pred_cov=output_cov, targ = gt_encoding, device=self.device, epoch=self.current_epoch)

        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        return optimizer

class VelocityVectorRegressor(pl.LightningModule):
    def __init__(self, learning_rate, input_dim, output_dim, group_sizes, intermediate_dim):
        super().__init__()
        self.save_hyperparameters()

        self.net = ResNet1D(
                BasicBlock1D, input_dim, output_dim, group_sizes, intermediate_dim
            )
        
        self.learning_rate = learning_rate
        
    def training_step(self, batch, batch_idx):

        # training step
        meas_seg, gt_encoding = batch

        # forward + backward + optimize
        outputs, output_cov = self.net(meas_seg)

        # compute and return loss
        loss = loss_mse_into_NLL(pred = outputs, pred_cov=output_cov, targ = gt_encoding, device=self.device, epoch=self.current_epoch)

        self.log("training_loss", loss, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):

        # training step
        meas_seg, gt_encoding = batch

        # forward + backward + optimize
        outputs, output_cov = self.net(meas_seg)

        # compute and return loss
        loss = loss_mse_into_NLL(pred = outputs, pred_cov=output_cov, targ = gt_encoding, device=self.device, epoch=self.current_epoch)

        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):

        # training step
        meas_seg, gt_encoding = batch

        # forward + backward + optimize
        outputs, output_cov = self.net(meas_seg)

        # compute and return loss
        loss = loss_mse_into_NLL(pred = outputs, pred_cov=output_cov, targ = gt_encoding, device=self.device, epoch=self.current_epoch)

        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        return optimizer