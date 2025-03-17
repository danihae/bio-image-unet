import os
import random
import time
from typing import Union, Tuple

import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import DataProcess
from .losses import *
from .multi_output_nested_unet import MultiOutputNestedUNet
from ..utils import init_weights, get_device


class Trainer:
    def __init__(self, dataset: DataProcess, num_epochs: int, network=MultiOutputNestedUNet, levels: int = 4,
                 batch_size: int = 4, lr: float = 1e-4, in_channels: int = 1, output_heads: Union[None, dict] = None,
                 n_filter: int = 64, deep_supervision: bool = False,
                 dilation: Union[bool, Tuple[int, int, int, int, int], Tuple[int, int, int, int]] = False,
                 val_split: float = 0.2, save_dir: str = './',
                 save_name: str = 'model.pt', save_iter: bool = False, load_weights: bool = False,
                 device: Union[torch.device, str] = 'auto'):
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.model = network(n_filter=n_filter, in_channels=in_channels, output_heads=output_heads, dilation=dilation,
                             deep_supervision=deep_supervision).to(self.device)
        self.model.apply(init_weights)
        self.data = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.levels = levels
        self.lr = lr
        self.best_loss = torch.tensor(float('inf'))
        self.save_iter = save_iter
        self.n_filter = n_filter
        self.dilation = dilation
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
        self.save_dir_val_result = save_dir + '/val_results/'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir_val_result, exist_ok=True)
        self.save_name = save_name
        self.params = {
            'optimizer': self.optimizer.state_dict(),
            'lr': self.lr,
            'n_filter': self.n_filter,
            'deep_supervision': deep_supervision,
            'dilation': self.dilation,
            'batch_size': self.batch_size,
            'augmentation': self.data.aug_factor,
            'clip_threshold': self.data.clip_threshold,
            'gauss_noise_lims': self.data.gauss_noise_lims,
            'shot_noise_lims': self.data.shot_noise_lims,
            'brightness_contrast': self.data.brightness_contrast,
            'random_rotate': self.data.random_rotate,
            'in_channels': in_channels,
            'output_heads': output_heads
        }
        if load_weights:
            self.state = torch.load(os.path.join(self.save_dir, self.save_name))
            self.model.load_state_dict(self.state['state_dict'])
            self.epoch_start = self.state['epoch']
        else:
            self.epoch_start = 0

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

                if x_i.dim() == 3:
                    x_i = x_i.unsqueeze(1)

                # Forward pass
                y_pred = self.model(x_i)

                # Handle deep supervision outputs
                total_loss = 0
                for name, config in self.output_heads.items():
                    target = y_i[name]
                    if target.dim() == 3:
                        target = target.unsqueeze(1)

                    if self.model.deep_supervision:
                        # Calculate loss for each deep supervision output
                        if self.levels == 3:
                            supervision_weights = [0.5, 0.75, 1.0]  # Weights for different supervision levels
                        elif self.levels == 4:
                            supervision_weights = [0.5, 0.75, 0.875, 1.0]
                        else:
                            raise ValueError(f'N = {self.levels} levels not valid. '
                                             f'Choose N=3 or N=4 according to network architecture.')
                        for level, weight in enumerate(supervision_weights, 1):
                            pred_key = f"{name}_{level}"
                            pred = y_pred[pred_key]
                            loss = self.loss_functions[name](pred, target)
                            total_loss += weight * self.loss_weights[name] * loss
                    else:
                        pred = y_pred[name]
                        loss = self.loss_functions[name](pred, target)
                        total_loss += self.loss_weights[name] * loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += total_loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch + self.epoch_start)

        elif mode == 'val':
            loss_list = []
            with torch.no_grad():
                for i, batch_i in tqdm(enumerate(self.val_loader), total=len(self.val_loader), unit='batch'):
                    x_i = batch_i['image'].to(self.device, non_blocking=True)
                    y_i = {key: batch_i[key].to(self.device, non_blocking=True) for key in self.output_heads}

                    if x_i.dim() == 3:
                        x_i = x_i.unsqueeze(1)

                    # Forward pass
                    y_pred = self.model(x_i)

                    # Compute combined loss
                    total_loss = 0
                    for name in self.output_heads:
                        target = y_i[name]
                        if target.dim() == 3:
                            target = target.unsqueeze(1)

                        if hasattr(self.model, 'deep_supervision') and self.model.deep_supervision:
                            supervision_weights = [0.5, 0.75, 1.0]
                            for level, weight in enumerate(supervision_weights, 1):
                                pred_key = f"{name}_{level}"
                                pred = self._apply_activation(y_pred[pred_key], self.activations.get(name))
                                loss = self.loss_functions[name](pred, target)
                                total_loss += weight * self.loss_weights[name] * loss
                        else:
                            pred = self._apply_activation(y_pred[name], self.activations.get(name))
                            loss = self.loss_functions[name](pred, target)
                            total_loss += self.loss_weights[name] * loss

                    loss_list.append(total_loss.detach())

            val_loss = torch.stack(loss_list).mean()
            self.writer.add_scalar('Loss/val', val_loss.item(), epoch + self.epoch_start)
            return val_loss

        torch.cuda.empty_cache()

    def plot_images(self, epoch, idx, x_i, y_pred, y_true, output_heads):
        x_i_np = x_i.cpu().numpy().squeeze()

        if hasattr(self.model, 'deep_supervision') and self.model.deep_supervision:
            num_heads = len(output_heads)
            num_levels = self.levels  # Number of deep supervision levels
            fig, axes = plt.subplots(2 + num_levels, num_heads + 1, figsize=(12, 12), dpi=300)

            # Plot input image
            for row in range(2 + num_levels):
                axes[row, 0].imshow(x_i_np, cmap='gray')
                axes[row, 0].set_title('Input' if row == 0 else '')
                axes[row, 0].set_xticks([])
                axes[row, 0].set_yticks([])

            for i, name in enumerate(output_heads):
                cmap = 'viridis' if name in ['length', 'orientation'] else 'gray'

                # Plot final prediction
                final_pred = y_pred[f"{name}"].cpu().numpy().squeeze()
                if len(final_pred.shape) == 3:
                    final_pred = final_pred[0]
                axes[0, i + 1].imshow(final_pred, cmap=cmap)
                axes[0, i + 1].set_title(f'{name} (Final)')
                axes[0, i + 1].set_xticks([])
                axes[0, i + 1].set_yticks([])

                # Plot ground truth
                y_true_np = y_true[name].cpu().numpy().squeeze()
                if len(y_true_np.shape) == 3:
                    y_true_np = y_true_np[0]
                axes[1, i + 1].imshow(y_true_np, cmap=cmap)
                axes[1, i + 1].set_title(f'{name} (True)')
                axes[1, i + 1].set_xticks([])
                axes[1, i + 1].set_yticks([])

                # Plot intermediate supervision outputs
                for level in range(1, num_levels + 1):
                    pred_key = f"{name}_{level}"
                    pred = y_pred[pred_key].cpu().numpy().squeeze()
                    if len(pred.shape) == 3:
                        pred = pred[0]
                    axes[level + 1, i + 1].imshow(pred, cmap=cmap)
                    axes[level + 1, i + 1].set_title(f'Level {level}')
                    axes[level + 1, i + 1].set_xticks([])
                    axes[level + 1, i + 1].set_yticks([])
        else:
            # Original plotting code for non-deep supervision
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

            for i, name in enumerate(output_heads):
                cmap = 'viridis' if name in ['length', 'orientation'] else 'gray'

                # Plot prediction in top row
                pred_np = y_pred[name].cpu().numpy().squeeze()
                if len(pred_np.shape) == 3:
                    pred_np = pred_np[0]
                axes[0, i + 1].imshow(pred_np, cmap=cmap)
                axes[0, i + 1].set_title(f'{name} (Pred)')
                axes[0, i + 1].set_xticks([])
                axes[0, i + 1].set_yticks([])

                # Plot ground truth in bottom row
                true_np = y_true[name].cpu().numpy().squeeze()
                if len(true_np.shape) == 3:
                    true_np = true_np[0]
                axes[1, i + 1].imshow(true_np, cmap=cmap)
                axes[1, i + 1].set_title(f'{name} (True)')
                axes[1, i + 1].set_xticks([])
                axes[1, i + 1].set_yticks([])

        plt.suptitle(f'Epoch {epoch + self.epoch_start}, Sample {idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir_val_result + f'Epoch {epoch + self.epoch_start}, Sample {idx}.png')
        plt.close()

    def log_validation_images(self, epoch, num_images):
        with torch.no_grad():
            all_indices = list(range(len(self.val_loader.dataset)))
            random.seed(self.random_seed)
            selected_indices = random.sample(all_indices, min(num_images, len(all_indices)))

            for idx in selected_indices:
                batch_i = self.val_loader.dataset[idx]

                x_i = batch_i['image'].unsqueeze(0).to(self.device)
                if x_i.dim() == 3:
                    x_i = x_i.unsqueeze(1)

                y_i = {key: batch_i[key].unsqueeze(0).to(self.device) for key in self.output_heads}
                for key in y_i:
                    if y_i[key].dim() == 3:
                        y_i[key] = y_i[key].unsqueeze(1)

                # Forward pass
                y_pred = self.model(x_i)

                if hasattr(self.model, 'deep_supervision') and self.model.deep_supervision:
                    # Handle deep supervision outputs
                    for name in self.output_heads:
                        # Log input image
                        self.writer.add_image(f'Validation/{name}_input_{idx}', x_i[0], epoch)

                        # Log true target image
                        self.writer.add_image(f'Validation/{name}_true_{idx}', y_i[name][0], epoch)

                        # Log predictions for each supervision level
                        for level in range(1, self.levels + 1):
                            pred_key = f"{name}_{level}"
                            pred = self._apply_activation(y_pred[pred_key], self.activations.get(name))
                            self.writer.add_image(f'Validation/{name}_pred_level{level}_{idx}', pred[0],
                                                  epoch + self.epoch_start)
                else:
                    # Original behavior for non-deep supervision
                    y_pred = {name: self._apply_activation(pred, self.activations.get(name))
                              for name, pred in y_pred.items()}

                    for name, pred in y_pred.items():
                        self.writer.add_image(f'Validation/{name}_input_{idx}', x_i[0], epoch + self.epoch_start)
                        self.writer.add_image(f'Validation/{name}_pred_{idx}', pred[0], epoch + self.epoch_start)
                        self.writer.add_image(f'Validation/{name}_true_{idx}', y_i[name][0], epoch + self.epoch_start)

                # Plot images using matplotlib
                self.plot_images(epoch, idx, x_i[0], y_pred, y_i, self.output_heads)

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
                print('\nValidation loss improved from %s to %s - saving model state' % (
                    round(self.best_loss.item(), 5), round(val_loss.item(), 5)))
                self.state['best_loss'] = self.best_loss = val_loss
                torch.save(self.state, os.path.join(self.save_dir, self.save_name))

            self.log_validation_images(epoch=epoch, num_images=10)

            # Save model state after each epoch if required
            if self.save_iter:
                torch.save(self.state, os.path.join(self.save_dir, f'model_epoch_{epoch + self.epoch_start}.pt'))