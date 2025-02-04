import os
import cv2
import datetime
import torch
from tensorboardX import SummaryWriter

class ExperimentLogger:
    def __init__(self, logdir='experiments', exp_name="test", logfile='log.txt'):
        self.logdir = logdir
        self.exp_name = exp_name
        self.create_exp_path()
        self.writer = SummaryWriter(self.exp_path)


    def create_exp_path(self):
        # create experiment dir by appending date and time
        self.exp_path = os.path.join(self.logdir, self.exp_name)
        self.exp_path += datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.exp_path, exist_ok=True)
        self.model_path = os.path.join(self.exp_path, 'models')
        os.makedirs(self.model_path, exist_ok=True)


    def log(self, train_loss_info, val_loss_info, step):
        self.log_to_file(train_loss_info, val_loss_info, step)
        for split, loss_info in zip(['train', 'val'], [train_loss_info, val_loss_info]):
            for key, value in loss_info.diagnostics_info.items():
                self.writer.add_scalar(f"{split}/{key}", value, step)

    
    def log_to_file(self, train_loss_info, val_loss_info, step):
        with open(os.path.join(self.exp_path, 'log.txt'), 'a') as f:
            f.write(f"==================================================\n")
            f.write(f"Step: {step}\n")
            for split, loss_info in zip(['train', 'val'], [train_loss_info, val_loss_info]):
                f.write(f"  {split}")
                for key, value in loss_info.diagnostics_info.items():
                    f.write(f"\t\t{key}: \t\t\t{value}\n")
            f.write(f"==================================================\n")


    def log_scalar(self, key, value, step):
        self.writer.add_scalar(key, value, step)


    def save_model(self, model, optimizer, name, step):
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{self.model_path}/{name}_{step}.pth')
