import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import network.dido_preprocessor as dido_preprocessor
from metrics import classification_metrics

from datetime import datetime
from pytz import timezone

import logging

logging.basicConfig(level=logging.INFO)
import numpy as np

import signal
from functools import partial
import sys

import os
from os import path as osp

from network.resnet1d.resnet1d import ResNet1D, BasicBlock1D, compute_model_args
from network.resnet1d.loss import loss_mse

from network.modules import VelocityUnitVectorRegressor

# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot

def save_model(args, epoch, network, optimizer, dt, best_v_loss, best=False):
    if not best:
        model_path = osp.join(
            args.root_dir, args.checkpoint_dir, "checkpoint_latest.pt"
        )
        state_dict = {
            "model_state_dict": network.state_dict(),
            "epoch": epoch,
            "best_v_loss": best_v_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(state_dict, model_path)
        logging.info(f"Model saved to {model_path}")
    else:
        model_path = osp.join(
            args.root_dir,
            args.model_output,
            args.model_name + "_best_val_loss_" + dt + ".pt",
        )
        state_dict = {
            "model_state_dict": network.state_dict(),
            "epoch": epoch,
            "best_v_loss": best_v_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(state_dict, model_path)
        logging.info(f"Model saved to {model_path}")


def stop_signal_handler(args, epoch, network, optimizer, dt, best_v_loss, signal, frame):
    logging.info("-" * 30)
    logging.info("Early terminate")
    save_model(args, epoch, network, optimizer, dt, best_v_loss=best_v_loss, best=False)
    sys.exit()

def compute_model_args(args):
    net_config = {
        "in_dim": int((
            args.nominal_imu_frequency * args.inertial_window_length
        )
        // 32
        + 1)
    }
    return net_config

def train(args):

    # setup target device
    device = torch.device(args.device_name)

    # net config
    net_config = compute_model_args(args)

    net = ResNet1D(
            BasicBlock1D, args.input_dim, args.output_dim, [2, 2, 2, 2], net_config["in_dim"]
        ).to(device)
    
    logging.info(f'Network "resnet" loaded to device {device}')

    # declare optimizer and loss criterion

    # add weights to cross-entropy loss based on class distribution
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    # check if checkpoint_latest is present in checkpoint directory. if so, reload state dict from that point and proceed with training
    # if not, then empty-declare model

    checkpoint_dir = osp.join(
        args.root_dir, args.checkpoint_dir, "checkpoint_latest.pt"
    )

    if os.path.isfile(checkpoint_dir):
        checkpoints = torch.load(checkpoint_dir)
        start_epoch = checkpoints.get("epoch", 0)
        best_vloss = checkpoints.get("best_v_loss", 0)
        net.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info("Detected saved checkpoint. Starting from latest state")
    else:
        start_epoch = 0
        best_vloss = np.inf
        logging.info("No checkpoint detected, starting at zeroth epoch")

    if (best_vloss == 0):
        best_vloss = np.inf

    # setup tensorboard
    tz = timezone("EST")
    curr_dt = datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S")
    metrics_loc = osp.join(
        args.root_dir, args.tensorboard_dir, curr_dt, args.model_name + "_metrics"
    )
    model_loc = osp.join(
        args.root_dir, args.tensorboard_dir, curr_dt, args.model_name + "_model"
    )
    summary_writer = classification_metrics.TensorboardWriter(metrics_loc)
    model_writer = SummaryWriter(model_loc)

    # retrieve training and validation trajectories
    training_set_loc = osp.join(args.root_dir, args.training_set_loc)
    training_trajectories = dido_preprocessor.VelocityUnitVectorDataset(
        training_set_loc,
        args.inertial_window_length,
        args.sampling_frequency,
        1 / args.nominal_imu_frequency,
        args,
    )

    validation_set_loc = osp.join(args.root_dir, args.val_set_loc)
    validation_trajectories = dido_preprocessor.VelocityUnitVectorDataset(
        validation_set_loc,
        args.inertial_window_length,
        args.sampling_frequency,
        1 / args.nominal_imu_frequency,
        args,
    )

    # move training trajectories to target device

    # declare dataloaders for training and validation datasets
    train_dataloader = DataLoader(
        training_trajectories,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        validation_trajectories,
        batch_size=args.batch_size,
        shuffle=True,
    )

    logging.info(f"Number of train samples: {len(training_trajectories)}")
    logging.info(f"Number of val samples: {len(validation_trajectories)}")

    # model_writer.add_graph(model = net, input_to_model = training_trajectories.__getitem__(0)[0].unsqueeze(0).to(device))
    # model_writer.close()
    logging.info("Model architecture logged.")
    total_params = net.get_num_params()
    logging.info(f"Total number of parameters: {total_params}")

    # metrics collector
    classification_training_metrics = (
        classification_metrics.TrainingMetrics(
            batch_length=args.batch_size, device=device
        )
    )

    classification_validation_metrics = (
        classification_metrics.TrainingMetrics(
            batch_length=args.batch_size, device=device
        )
    )

    import time

    for epoch in range(
        start_epoch + 1, args.epochs
    ):  # loop over the dataset multiple times
        signal.signal(
            signal.SIGINT,
            partial(stop_signal_handler, args, epoch, net, optimizer, curr_dt, best_vloss),
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, net, optimizer, curr_dt, best_vloss),
        )

        net.train(True)

        for i, (meas_seg, gt_encoding) in enumerate(train_dataloader, 0):

            # retrieve next set of training data
            meas_seg, gt_encoding = meas_seg.to(device), gt_encoding.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(meas_seg)

            loss = loss_mse(pred = outputs, targ = gt_encoding)
            loss = torch.mean(loss)

            make_dot(loss, params=dict(net.named_parameters()))

            loss.backward()
            optimizer.step()

            # compute accuracy metrics from outputs and gt_encoding
            classification_training_metrics.update_metrics(
                outputs, gt_encoding, loss
            )

            print(f"completed training iteration {i} for epoch {epoch}")

        classification_metrics.publish_and_reset_metrics(
            summary_writer,
            classification_training_metrics,
            train_dataloader,
            epoch,
            "train",
        )

        # save model per epoch (to resume training, if required)
        save_model(args, epoch, net, optimizer, curr_dt, best_vloss)

        # turn model gradient tracking off
        if args.validate:
            net.train(False)

            # # run validation testing
            for i, v_data in enumerate(val_dataloader, 0):
                # retrieve next set of training data
                v_meas_seg, v_gt_encoding = v_data

                v_meas_seg, v_gt_encoding = v_meas_seg.to(device), v_gt_encoding.to(
                    device
                )

                v_outputs = net(v_meas_seg)

                v_loss = criterion(v_outputs, v_gt_encoding)

                classification_validation_metrics.update_metrics(
                    v_outputs, v_gt_encoding, v_loss
                )

            v_loss = classification_metrics.publish_and_reset_metrics(
                summary_writer,
                classification_validation_metrics,
                val_dataloader,
                epoch,
                "val",
            )

            # Track best performance, and save the model's state
            if v_loss < best_vloss:
                best_vloss = v_loss
                save_model(args, epoch, net, optimizer, curr_dt, best_vloss, best=True)

    print("Finished Training")
