import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from os import path as osp

from network.modules import VelocityUnitVectorRegressor, VelocityVectorRegressor

from network.dido_preprocessor import VelocityUnitVectorDataset

from lightning.pytorch.callbacks import ModelCheckpoint

from network.resnet1d.resnet1d import ResNet1D, BasicBlock1D, compute_model_args


# lightning training loop
def pl_train(args):
    # retrieve training and validation trajectories
    training_set_loc = osp.join(args.root_dir, args.training_set_loc)
    training_trajectories = VelocityUnitVectorDataset(
        training_set_loc,
        args.inertial_window_length,
        args.sampling_frequency,
        1 / args.nominal_imu_frequency,
        args,
    )

    validation_set_loc = osp.join(args.root_dir, args.val_set_loc)
    validation_trajectories = VelocityUnitVectorDataset(
        validation_set_loc,
        args.inertial_window_length,
        args.sampling_frequency,
        1 / args.nominal_imu_frequency,
        args,
    )

    # test_set_loc = osp.join(args.root_dir, args.test_set_loc)
    # test_trajectories = VelocityUnitVectorDataset(
    #     test_set_loc,
    #     args.inertial_window_length,
    #     args.sampling_frequency,
    #     1 / args.nominal_imu_frequency,
    #     args,
    # )

    # declare dataloaders for training and validation datasets
    train_dataloader = DataLoader(
        training_trajectories,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=24,
    )

    val_dataloader = DataLoader(
        validation_trajectories,
        batch_size=args.batch_size,
        num_workers=24,
    )

    # test_dataloader = DataLoader(
    #     test_trajectories, batch_size=args.batch_size, num_workers=24
    # )

    # declare model
    net_config = compute_model_args(args)

    if (args.train_raw_velocity):
        regressor = VelocityVectorRegressor(
            args.target_learning_rate,
            args.input_dim,
            args.output_dim,
            [
                args.residual_block_depth,
                args.residual_block_depth,
                args.residual_block_depth,
                args.residual_block_depth,
            ],
            net_config["in_dim"],
        )
    else:
        regressor = VelocityUnitVectorRegressor(
            args.target_learning_rate,
            args.input_dim,
            args.output_dim,
            [
                args.residual_block_depth,
                args.residual_block_depth,
                args.residual_block_depth,
                args.residual_block_depth,
            ],
            net_config["in_dim"],
        )

    print(f"Total number of parameters : {regressor.net.get_num_params()}")

    # saves top-K checkpoints based on "val_loss" metric
    val_loss_cb = ModelCheckpoint(
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        dirpath="lightning_logs/validation_checkpoints/",
        filename= args.model_name + "_best_val_loss",
    )
    latest_epoch_cb = ModelCheckpoint(
        dirpath="lightning_logs/latest_epoch_checkpoints/",
        filename=args.model_name + "_latest_epoch",
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[val_loss_cb, latest_epoch_cb],
        max_epochs=args.epochs,
    )
    trainer.fit(
        model=regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # trainer.test(model=regressor, dataloaders=test_dataloader)
