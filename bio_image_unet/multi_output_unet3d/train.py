import os
import time
from typing import Union

import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .losses import *
from .multi_output_unet3d import MultiOutputUnet3D
from ..utils import init_weights, get_device


class Trainer:
    """
    Class for training of neural network. Creates trainer object, training is started with .start().

    Parameters
    ----------
    dataset
        Training data, object of PyTorch Dataset class
    output_heads
        Dictionary of output heads.
    num_epochs : int
        Number of training epochs
    network
        Network class (Default Unet)
    use_interpolation : bool
        Whether to use interpolation in decoder instead of transpose convolution.
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    in_channels : int
        Number of input channels
    n_filter : int
        Number of convolutional filters in first layer
    val_split : float
        Validation split
    save_dir : str
        Path of directory to save trained networks
    save_name : str
        Base name for saving trained networks
    save_iter : bool
        If True, network state is save after each epoch
    load_weights : str, optional
        If not None, network state is loaded before training
    loss_function : str
        Loss function ('BCEDice', 'Tversky' or 'logcoshTversky')
    loss_params : Tuple[float, float]
        Parameter of loss function, depends on chosen loss function
    device : torch.device or str, optional
        Device to run the pytorch model on, defaults to 'auto', which selects CUDA or MPS if available.
    """

    def __init__(self, dataset, output_heads, num_epochs, network=MultiOutputUnet3D, use_interpolation=False,
                 batch_size=4, lr=1e-3, in_channels=1, n_filter=64, dilation=1, val_split=0.2,
                 save_dir='./', save_name='model.pt', save_iter=False, load_weights=False, loss_function='BCEDice',
                 loss_params=(0.5, 0.5), time_loss_weight=0.1, device: Union[torch.device, str] = 'auto'):

        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)
        self.network = network
        self.model = network(n_filter=n_filter, in_channels=in_channels, output_heads=output_heads,
                             use_interpolation=use_interpolation).to(self.device)
        self.model.apply(init_weights)
        self.data = dataset
        self.output_heads = output_heads
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.best_loss = torch.tensor(float('inf'))
        self.save_iter = save_iter
        self.loss_function = loss_function
        self.loss_params = loss_params
        self.time_loss_weight = time_loss_weight
        self.n_filter = n_filter
        self.in_channels = in_channels
        self.use_interpolation = use_interpolation

        # Define loss functions for each head
        self.loss_functions = {
            name: self._get_loss_function(config['loss']) for name, config in self.output_heads.items()
        }

        # Define activations for each head
        self.activations = {
            name: config.get('activation', None) for name, config in self.output_heads.items()
        }

        # Define weights for each head
        self.loss_weights = {
            name: config.get('weight', 1.0) for name, config in self.output_heads.items()
        }

        # split training and validation data
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        self.dim = dataset.dim_out
        train_data, val_data = random_split(dataset, [num_train, num_val])
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        if loss_function == 'BCEDice':
            self.criterion = BCEDiceLoss(loss_params[0], loss_params[1])
        elif loss_function == 'Tversky':
            self.criterion = TverskyLoss(loss_params[0], loss_params[1])
        elif loss_function == 'logcoshTversky':
            self.criterion = logcoshTverskyLoss(loss_params[0], loss_params[1])
        elif loss_function == 'BCEDiceTemporalLoss':
            self.criterion = BCEDiceTemporalLoss(loss_params=loss_params)
        else:
            raise ValueError(f'Loss "{loss_function}" not defined!')
        self.criterion_time = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.2)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name = save_name
        self.params = {'optimizer': self.optimizer.state_dict(),
                       'lr': self.lr,
                       'loss_function': self.loss_function,
                       'loss_params': self.loss_params,
                       'time_loss_weight': self.time_loss_weight,
                       'n_filter': self.n_filter,
                       'use_interpolation': use_interpolation,
                       'dilation': dilation, 'batch_size': self.batch_size,
                       'augmentation': self.data.aug_factor,
                       'clip_threshold': self.data.clip_threshold,
                       'scale_limit': self.data.scale_limit,
                       'rotate_limit': self.data.rotate_limit,
                       'gauss_noise_lims': self.data.gauss_noise_lims,
                       'shot_noise_lims': self.data.shot_noise_lims,
                       'blur_limit': self.data.blur_limit,
                       'random_rotate': self.data.random_rotate,
                       'brightness_contrast': self.data.brightness_contrast,
                       'in_channels': in_channels,
                       'output_heads': output_heads}
        if load_weights:
            self.state = torch.load(os.path.join(self.save_dir, self.save_name))
            self.model.load_state_dict(self.state['state_dict'])
            self.epoch_start = self.state['epoch']
        else:
            self.epoch_start = 0

    @staticmethod
    def _get_loss_function(loss_name):
        if loss_name == 'BCEDiceLoss':
            return BCEDiceLoss(1, 1)
        elif loss_name == 'DiceLoss':
            return BCEDiceLoss(0, 1)
        elif loss_name == 'TverskyLoss':
            return TverskyLoss()
        elif loss_name == 'logcoshTverskyLoss':
            return logcoshTverskyLoss()
        elif loss_name == 'BCEDiceTemporalLoss':
            return BCEDiceTemporalLoss()
        else:
            raise ValueError(f'Loss "{loss_name}" not defined!')

    @staticmethod
    def _apply_activation(x, activation):
        if activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'tanh':
            return torch.tanh(x)
        elif activation == 'relu':
            return torch.relu(x)
        elif activation == 'softmax':
            return torch.softmax(x, dim=1)
        return x

    def __iterate(self, epoch, mode):
        if mode == 'train':
            running_loss = 0.0
            for i, batch_i in tqdm(enumerate(self.train_loader), total=len(self.train_loader), unit='batch'):
                x_i = batch_i['volume'].to(self.device, non_blocking=True)
                y_i = {key: batch_i[key].to(self.device, non_blocking=True) for key in self.output_heads}

                if x_i.dim() == 4:
                    x_i = x_i.unsqueeze(1)

                # Forward pass
                y_pred = self.model(x_i)

                # Handle outputs
                total_loss = 0
                for name, config in self.output_heads.items():
                    target = y_i[name]
                    if target.dim() == 4:
                        target = target.unsqueeze(1)
                    pred = y_pred[name]
                    loss = self.loss_functions[name](pred, target)
                    total_loss += self.loss_weights[name] * loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        elif mode == 'val':
            loss_list = []
            with torch.no_grad():
                for i, batch_i in tqdm(enumerate(self.val_loader), total=len(self.val_loader), unit='batch'):
                    x_i = batch_i['volume'].to(self.device, non_blocking=True)
                    y_i = {key: batch_i[key].to(self.device, non_blocking=True) for key in self.output_heads}

                    if x_i.dim() == 4:
                        x_i = x_i.unsqueeze(1)

                    # Forward pass
                    y_pred = self.model(x_i)

                    # Compute combined loss
                    total_loss = 0
                    for name in self.output_heads:
                        target = y_i[name]

                        if target.dim() == 4:
                            target = target.unsqueeze(1)
                        pred = self._apply_activation(y_pred[name], self.activations.get(name))
                        loss = self.loss_functions[name](pred, target)
                        total_loss += self.loss_weights[name] * loss

                    loss_list.append(total_loss.detach())

            val_loss = torch.stack(loss_list).mean()
            return val_loss

        torch.cuda.empty_cache()

    def start(self):
        for epoch in range(self.num_epochs):
            start_time = time.time()  # Start time of epoch

            # Training phase
            self.__iterate(epoch, 'train')

            # Update state dictionary
            self.state = {
                'epoch': epoch + self.epoch_start,
                'epoch_start': self.epoch_start,
                'best_loss': self.best_loss,
                'state_dict': self.model.state_dict()
            }
            self.state.update(self.params)

            # Validation phase
            with torch.no_grad():
                val_loss = self.__iterate(epoch, 'val')
                self.scheduler.step(val_loss)

            # Calculate time taken for epoch
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)

            print(f"\nEpoch {epoch} completed in {time_taken} seconds.")

            # Check for improvement in validation loss
            if val_loss < self.best_loss:
                print(
                    f'\nValidation loss improved from {self.best_loss.item():.5f} to {val_loss.item():.5f} - saving model state')
                self.state['best_loss'] = self.best_loss = val_loss
                torch.save(self.state, os.path.join(self.save_dir, self.save_name))
            else:
                print(f'\nValidation loss did not improve from {self.best_loss.item():.5f}')

            # Save model state after each epoch if required
            if self.save_iter:
                torch.save(self.state, os.path.join(self.save_dir, f'model_epoch_{epoch + self.epoch_start}.pt'))