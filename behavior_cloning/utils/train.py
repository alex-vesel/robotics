import sys
from time import monotonic
from tqdm import tqdm

import torch
import numpy as np

from behavior_cloning.configs.nn_config import *


def train_model(train_loader, val_loader, model, configs, optimizer, logger, num_epochs=100, log_steps=10, device='cpu'):
    num_steps = 0
    train_losses = []
    for epoch in range(num_epochs):
        start_time = monotonic()

        model.train()
        for batch in tqdm(train_loader):
            for key in batch:
                batch[key] = batch[key].to(device)

            output = model(batch['depth_frame'], batch['wrist_frame'], batch['angle'])

            # print(output['angle_regression'][0]*30, batch['delta_angle'][0])
            # print(output['motor_activation'][0])

            subtask_losses = {}
            total_loss = 0
            for config in configs:
                subtask_loss = config.get_loss(output[config.name], batch[config.gt_key], split='train', weight=batch['weight'])
                if torch.isnan(subtask_loss):
                    continue
                subtask_losses[config.name] = subtask_loss
                total_loss += subtask_loss

            if total_loss > 0:
                subtask_losses['total_loss'] = total_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_losses.append(subtask_losses)
            num_steps += 1

            if len(train_losses) >= log_steps:
                # concatenate train_losses
                for key in train_losses[0]:
                    avg_loss = np.mean([float(loss[key].detach().cpu().numpy()) for loss in train_losses if key in loss])
                    logger.log_scalar(f'train/{key}', avg_loss, num_steps)
                train_losses = []

        end_time = monotonic()

        logger.log_scalar('utils/elapsed_time', end_time - start_time, epoch)
        logger.log_scalar('utils/sec_per_sample', (end_time - start_time) / len(train_loader.dataset), epoch)
        logger.log_scalar('utils/fps', len(train_loader.dataset) / (end_time - start_time), epoch)
        logger.log_scalar('utils/lr', optimizer.param_groups[0]['lr'], epoch)

        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(device)

                output = model(batch['depth_frame'], batch['wrist_frame'], batch['angle'])

                subtask_losses = {}
                total_loss = 0
                for config in configs:
                    subtask_loss = config.get_loss(output[config.name], batch[config.gt_key])
                    if torch.isnan(subtask_loss):
                        continue
                    subtask_losses[config.name] = subtask_loss
                    total_loss += subtask_loss

                if total_loss > 0:
                    subtask_losses['total_loss'] = total_loss

                val_losses.append(subtask_losses)

        # concatenate val_losses
        for key in val_losses[0]:
            avg_loss = np.mean([float(loss[key].detach().cpu().numpy()) for loss in val_losses if key in loss])
            logger.log_scalar(f'val/{key}', avg_loss, num_steps)

        # save model
        if epoch % SAVE_EPOCHS == 0:
            logger.save_model(model, optimizer, 'model', num_steps)