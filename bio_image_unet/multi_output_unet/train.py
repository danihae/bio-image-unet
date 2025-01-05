import os
import random
from typing import Union

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import *
from .multi_output_unet import MultiOutputUnet
from ..utils import init_weights, get_device


class Trainer:
    def __init__(self, dataset, num_epochs, network=MultiOutputUnet, batch_size=4, lr=1e-4, in_channels=1,
                 output_heads=None, n_filter=64,
                 val_split=0.2, save_dir='./', save_name='model.pth',
                 save_iter=False, load_weights=False,
                 device: Union[torch.device, str] = 'auto'):
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.network = network
        self.model = network(n_filter=n_filter, in_channels=in_channels, output_heads=output_heads).to(self.device)
        self.model.apply(init_weights)
        self.data = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.best_loss = torch.tensor(float('inf'))
        self.save_iter = save_iter
        self.n_filter = n_filter
        self.in_channels = in_channels
        self.output_heads = output_heads

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

        # Initialize channel weights for each output head
        self.channel_weights = {name: torch.ones(config['channels']) for name, config in self.output_heads.items()}

        # Split training and validation data
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        self.dim = dataset.dim_out
        train_data, val_data = random_split(dataset, [num_train, num_val])
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.2)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name = save_name
        self.params = {
            'optimizer': self.optimizer.state_dict(),
            'lr': self.lr,
            'n_filter': self.n_filter,
            'batch_size': self.batch_size,
            'augmentation': self.data.aug_factor,
            'clip_threshold': self.data.clip_threshold,
            'gauss_noise_lims': self.data.gauss_noise_lims,
            'shot_noise_lims': self.data.shot_noise_lims,
            'brightness_contrast': self.data.brightness_contrast,
            'in_channels': in_channels,
            'output_heads': output_heads
        }
        if load_weights:
            self.state = torch.load(os.path.join(self.save_dir, self.save_name))
            self.model.load_state_dict(self.state['state_dict'])

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))

        # Set a fixed random seed for reproducibility
        self.random_seed = 42
        random.seed(self.random_seed)

    @staticmethod
    def _get_loss_function(loss_name):
        if loss_name == 'BCEDiceLoss':
            return BCEDiceLoss()
        elif loss_name == 'DiceLoss':
            return BCEDiceLoss(bce_weight=0, dice_weight=1)
        elif loss_name == 'TverskyLoss':
            return TverskyLoss()
        elif loss_name == 'logcoshTverskyLoss':
            return logcoshTverskyLoss()
        elif loss_name == 'MSELoss':
            return MSELoss()
        elif loss_name == 'MAELoss':
            return MAELoss()
        elif loss_name == 'HuberLoss':
            return HuberLoss()
        elif loss_name == 'DistanceGradientLoss':
            return DistanceGradientLoss()
        elif loss_name == 'WeightedDistanceGradientLoss':
            return WeightedDistanceGradientLoss()
        elif loss_name == 'WeightedVectorFieldLoss':
            return WeightedVectorFieldLoss()
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
                x_i = batch_i['image'].to(self.device, non_blocking=True)
                y_i = {key: batch_i[key].to(self.device, non_blocking=True) for key in self.output_heads}

                # Ensure x_i has a channel dimension
                if x_i.dim() == 3:  # If shape is (batch_size, height, width)
                    x_i = x_i.unsqueeze(1)  # Add channel dimension

                # Forward pass
                y_pred = self.model(x_i)

                # Apply activations
                y_pred = {name: self._apply_activation(pred, self.activations.get(name)) for name, pred in
                          y_pred.items()}

                # Compute combined loss
                total_loss = 0
                for name, pred in y_pred.items():
                    target = y_i[name]

                    # Ensure target has a channel dimension
                    if target.dim() == 3:  # If shape is (batch_size, height, width)
                        target = target.unsqueeze(1)  # Add channel dimension

                    loss_fn = self.loss_functions[name]
                    loss = loss_fn(pred, target)
                    total_loss += self.loss_weights[name] * loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += total_loss.item()
            avg_train_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)

        elif mode == 'val':
            loss_list = []
            with torch.no_grad():
                for i, batch_i in tqdm(enumerate(self.val_loader), total=len(self.val_loader), unit='batch'):
                    x_i = batch_i['image'].to(self.device, non_blocking=True)
                    y_i = {key: batch_i[key].to(self.device, non_blocking=True) for key in self.output_heads}

                    # Ensure x_i has a channel dimension
                    if x_i.dim() == 3:  # If shape is (batch_size, height, width)
                        x_i = x_i.unsqueeze(1)  # Add channel dimension

                    # Forward pass
                    y_pred = self.model(x_i)

                    # Apply activations
                    y_pred = {name: self._apply_activation(pred, self.activations.get(name)) for name, pred in
                              y_pred.items()}

                    # Compute combined loss
                    total_loss = 0
                    for name, pred in y_pred.items():
                        target = y_i[name]
                        # Ensure target has a channel dimension
                        if target.dim() == 3:  # If shape is (batch_size, height, width)
                            target = target.unsqueeze(1)  # Add channel dimension

                        loss_fn = self.loss_functions[name]
                        loss = loss_fn(pred, target)
                        total_loss += self.loss_weights[name] * loss
                    loss_list.append(total_loss.detach())

            val_loss = torch.stack(loss_list).mean()
            self.writer.add_scalar('Loss/val', val_loss.item(), epoch)
            return val_loss

        torch.cuda.empty_cache()

    @staticmethod
    def plot_images(epoch, idx, x_i, y_pred, y_true, output_heads):
        # Convert tensors to numpy arrays for plotting
        x_i_np = x_i.cpu().numpy().squeeze()
        y_pred_np = {name: pred.cpu().numpy().squeeze() for name, pred in y_pred.items()}
        y_true_np = {name: true.cpu().numpy().squeeze() for name, true in y_true.items()}

        # Plot input image, predicted, and true target images
        num_heads = len(output_heads)
        fig, axes = plt.subplots(2, num_heads + 1, figsize=(12, 8), dpi=300)

        # Plot input image in both rows
        axes[0, 0].imshow(x_i_np, cmap='gray')
        axes[0, 0].set_title('Input')
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[1, 0].imshow(x_i_np, cmap='gray')
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])

        # Plot each head's prediction and true target
        for i, name in enumerate(output_heads):
            if name in ['length', 'orientation']:
                cmap = 'viridis'
            else:
                cmap = 'gray'

            # Plot prediction in top row
            if len(y_pred_np[name].shape) == 3:
                axes[0, i + 1].imshow(y_pred_np[name][0], cmap=cmap)
            else:
                axes[0, i + 1].imshow(y_pred_np[name], cmap=cmap)
            axes[0, i + 1].set_title(f'{name} (Pred)')
            axes[0, i + 1].set_xticks([])
            axes[0, i + 1].set_yticks([])

            # Plot ground truth in bottom row
            if len(y_true_np[name].shape) == 3:
                axes[1, i + 1].imshow(y_true_np[name][0], cmap=cmap)
            else:
                axes[1, i + 1].imshow(y_true_np[name], cmap=cmap)
            axes[1, i + 1].set_title(f'{name} (True)')
            axes[1, i + 1].set_xticks([])
            axes[1, i + 1].set_yticks([])

        plt.suptitle(f'Epoch {epoch}, Sample {idx}')
        plt.tight_layout()
        plt.show()

    def log_validation_images(self, epoch, num_images):
        with torch.no_grad():
            # Collect all validation data indices
            all_indices = list(range(len(self.val_loader.dataset)))

            # Randomly select a subset of indices with a fixed seed
            random.seed(self.random_seed)
            selected_indices = random.sample(all_indices, min(num_images, len(all_indices)))

            for idx in selected_indices:
                # Get the batch corresponding to the selected index
                batch_i = self.val_loader.dataset[idx]

                x_i = batch_i['image'].unsqueeze(0).to(self.device)  # Add batch dimension
                # Ensure x_i has a channel dimension
                if x_i.dim() == 3:  # If shape is (1, height, width)
                    x_i = x_i.unsqueeze(1)  # Add channel dimension

                y_i = {key: batch_i[key].unsqueeze(0).to(self.device) for key in self.output_heads}
                for key in y_i:
                    # Ensure target has a channel dimension
                    if y_i[key].dim() == 3:  # If shape is (1, height, width)
                        y_i[key] = y_i[key].unsqueeze(1)  # Add channel dimension

                # Forward pass
                y_pred = self.model(x_i)

                # Apply activations
                y_pred = {name: self._apply_activation(pred, self.activations.get(name)) for name, pred in
                          y_pred.items()}

                for name, pred in y_pred.items():
                    # Log input image
                    self.writer.add_image(f'Validation/{name}_input_{idx}', x_i[0], epoch)

                    # Log predicted image
                    self.writer.add_image(f'Validation/{name}_pred_{idx}', pred[0], epoch)

                    # Log true target image
                    self.writer.add_image(f'Validation/{name}_true_{idx}', y_i[name][0], epoch)

                # Plot images using matplotlib
                self.plot_images(epoch, idx, x_i[0], y_pred, y_i, self.output_heads)

    def start(self):
        for epoch in range(self.num_epochs):
            # Training phase
            self.__iterate(epoch, 'train')

            # Update state dictionary
            self.state = {
                'epoch': epoch,
                'best_loss': self.best_loss,
                'state_dict': self.model.state_dict()
            }
            self.state.update(self.params)

            # Validation phase
            with torch.no_grad():
                val_loss = self.__iterate(epoch, 'val')
                self.scheduler.step(val_loss)

            # Check for improvement in validation loss
            if val_loss < self.best_loss:
                print('\nValidation loss improved from %s to %s - saving model state' % (
                    round(self.best_loss.item(), 5), round(val_loss.item(), 5)))
                self.state['best_loss'] = self.best_loss = val_loss
                torch.save(self.state, os.path.join(self.save_dir, self.save_name))

            self.log_validation_images(epoch=epoch, num_images=6)

            # Save model state after each epoch if required
            if self.save_iter:
                torch.save(self.state, os.path.join(self.save_dir, f'model_epoch_{epoch}.pth'))
