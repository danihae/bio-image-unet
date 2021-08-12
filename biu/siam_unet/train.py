import os, glob, re, shutil
import subprocess
from sys import platform
import time
from numpy.lib.function_base import diff
import random

import tifffile
import numpy as np
from skimage import morphology, transform
from barbar import Bar
from tifffile.tifffile import TiffFile
from tqdm import tqdm as tqdm

import torch
from torch import nn as nn, flatten
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from . import losses

from .helpers.util import write_info_file
from .helpers.__md5sum__ import md5sum, md5sum_folder
from .siam_unet import Siam_UNet
from .predict import Predict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, dataset, num_epochs, batch_size=4, lr=1e-3, n_filter=64, val_split=0.2,
                 save_dir='./', save_name='model.pth', save_iter=False, loss_function=losses.logcoshDiceLoss, load_weights=False):
        """Trainer for Siam-UNet

        Args:
            dataset (DataProcess): data loader for the training dataset
            num_epochs (int): intended number of epochs to train. A good number for the BCE loss function is 300.
            batch_size (int, optional): batch size for training. Defaults to 4.
            lr (float, optional): learning rate. Defaults to 1e-3.
            n_filter (int, optional): number of filters. Defaults to 64.
            val_split (float, optional): which proportion of training data to be split into the validation data category. Defaults to 0.2.
            save_dir (str, optional): where to save the model(s). Defaults to './'.
            save_name (str, optional): the name of the saved model. Only used when save_iter is false. Defaults to 'model.pth'.
            save_iter (bool, optional): saves the model at each iteration under the name model_epoch_{iter}.pth. Defaults to False.
            loss_function: losses.logcoshDiceLoss or losses.BCEDiceLoss. Defaults to losses.logcoshDiceLoss.
        """

        self.model = Siam_UNet(n_filter=n_filter).to(device)

        self.data = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.best_loss = torch.tensor(float('inf'))
        self.save_iter = save_iter
        # split training and validation data
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        self.dim = dataset.dim_out
        train_data, val_data = random_split(dataset, [num_train, num_val])
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.criterion = loss_function(1, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, factor=0.1)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name = save_name
        write_info_file((save_dir + '/' + save_name) + '.info.txt', f'Mode: Train with Siam_UNet\nOutfile name:{(save_dir + "/" + save_name)}\nDataset folder: {dataset}\nIntended Epochs: {num_epochs}')

        if load_weights:
            self.state = torch.load(self.save_dir + '/' + self.save_name)
            self.model.load_state_dict(self.state['state_dict'])

    def iterate(self, epoch, mode):
        if mode == 'train':
            print('\nStarting training epoch %s ...' % epoch)
            for i, batch_i in enumerate(Bar(self.train_loader)):
                x_i = batch_i['image'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                prev_x_i = batch_i['prev_image'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                y_i = batch_i['mask'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.model(x_i, prev_x_i)

                # Compute and print loss
                loss = self.criterion(y_pred, y_i)

                if random.random() > 0.99:
                    print('Mean, min, max', torch.mean(y_i), torch.min(y_i), torch.max(y_i))


                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        elif mode == 'val':
            loss_list = []
            print('\nStarting validation epoch %s ...' % epoch)
            with torch.no_grad():
                for i, batch_i in enumerate(Bar(self.val_loader)):
                    x_i = batch_i['image'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                    prev_x_i = batch_i['prev_image'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                    y_i = batch_i['mask'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred = self.model(x_i, prev_x_i)
                    loss = self.criterion(y_pred, y_i)
                    loss_list.append(loss.detach())
            val_loss = torch.stack(loss_list).mean()
            return val_loss

        torch.cuda.empty_cache()

    def start(self, test_data_path=None, result_path=None):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, 'train')
            self.state = {
                'epoch': epoch,
                'best_loss': self.best_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr': self.lr,
                'augmentation': self.data.aug_factor,
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, 'val')
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print('\nValidation loss improved from %s to %s - saving model state' % (
                    round(self.best_loss.item(), 5), round(val_loss.item(), 5)))
                self.state['best_loss'] = self.best_loss = val_loss
                torch.save(self.state, self.save_dir + '/' + self.save_name)
            if self.save_iter:
                torch.save(self.state, self.save_dir + '/' + f'model_epoch_{epoch}.pth')

            if test_data_path is not None:
                files = glob.glob(test_data_path + '*.tif')
                for i, file in enumerate(files):
                    Predict(file, result_path + os.path.basename(file) + f'epoch_{epoch}.tif', self.save_dir + '/' +
                            f'model_epoch_{epoch}.pth', resize_dim=(512, 512), invert=False)

