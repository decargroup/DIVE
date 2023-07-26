import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.classification import BinaryF1Score, MulticlassAccuracy
from torchmetrics.classification import BinaryConfusionMatrix

import logging

class TensorboardWriter():
    def __init__(self, log_dir):

        self.writer = SummaryWriter(log_dir)
    
    def write_scalar(self, label, val, i, mode):
        scalar_label = (label + "_" + mode)
        self.writer.add_scalar(scalar_label, val, i)

    def write_image(self, label, img, i, mode):
        image_label = (label + "_" + mode)
        self.writer.add_figure(image_label, img, i)

class TrainingMetrics:
    def __init__(self, batch_length, device):

        # running loss metric
        self.running_loss = 0

        self.batch_length = batch_length

        # declare device
        self.device = device

    def update_metrics(self, outputs, gt_encoding, loss):

        self.running_loss += loss.item()

    # reset performance metrics completely on epoch change
    def reset_metrics(self):
        self.running_loss = 0

def publish_and_reset_metrics(scalar_writer : TensorboardWriter, planar_classification_metrics : TrainingMetrics, train_dataloader, epoch, mode):

    # log the running loss
    scalar_writer.write_scalar(
        "loss",
        planar_classification_metrics.running_loss / planar_classification_metrics.batch_length,
        epoch * len(train_dataloader) + planar_classification_metrics.batch_length, mode
    )
    logging.info(
        f"[{epoch + 1}, {planar_classification_metrics.batch_length:5d}] loss: {planar_classification_metrics.running_loss / planar_classification_metrics.batch_length:.3f}"
    )

    planar_classification_metrics.reset_metrics()
