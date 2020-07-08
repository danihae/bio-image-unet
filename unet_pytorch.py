import os, glob, re, shutil
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from barbar import Bar
from tqdm import tqdm as tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from albumentations import (
    ShiftScaleRotate, GaussNoise, GaussianBlur,
    RandomBrightnessContrast, Flip, Compose, ElasticTransform)

torch.set_default_dtype(torch.float32)
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    print('No CUDA device found. Using CPU instead ...')

print('Import U-Net package ...')


class DataProcess(Dataset):
    def __init__(self, source_dir, file_ext='.tif', dim_out=(256, 256), aug_factor=10, data_path='./data/',
                 dilate_mask=1, val_split=0.1,
                 CREATE=False):
        self.source_dir = source_dir
        self.file_ext = file_ext
        self.CREATE = CREATE
        self.data_path = data_path
        self.dim_out = dim_out
        self.aug_factor = aug_factor
        self.dilate_mask = dilate_mask
        self.val_split = val_split
        self.mode = 'train'

        self.make_dirs()

        if CREATE:
            self.move_and_edit()
            self.merge_images()
            self.split()
            self.augment()

    def make_dirs(self):
        self.image_path = './data/image/'
        self.mask_path = './data/mask/'
        self.npy_path = './data/npydata/'
        self.merge_path = self.data_path + '/merge/'
        self.split_merge_path = self.data_path + '/split/merge/'
        self.split_image_path = self.data_path + '/split/image/'
        self.split_mask_path = self.data_path + '/split/mask/'
        self.aug_image_path = './data/augmentation/aug_image/'
        self.aug_mask_path = './data/augmentation/aug_mask/'

        # delete old files
        if self.CREATE:
            try:
                shutil.rmtree('./data/')
            except:
                pass
        # make folders
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.npy_path, exist_ok=True)
        os.makedirs(self.merge_path, exist_ok=True)
        os.makedirs(self.split_merge_path, exist_ok=True)
        os.makedirs(self.split_image_path, exist_ok=True)
        os.makedirs(self.split_mask_path, exist_ok=True)
        os.makedirs(self.aug_image_path, exist_ok=True)
        os.makedirs(self.aug_mask_path, exist_ok=True)

    def move_and_edit(self):
        # create image data
        files_image = glob.glob(self.source_dir[0] + '*' + self.file_ext)
        for file_i in files_image:
            img_i = tifffile.imread(file_i)
            # clip and normalize (0,255)
            img_i = np.clip(img_i, a_min=np.percentile(img_i, 0.2), a_max=np.percentile(img_i, 99.8))
            img_i = img_i - np.min(img_i)
            img_i = img_i / np.max(img_i) * 255
            img_i = img_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            tifffile.imsave(self.image_path + save_i + '.tif', img_i)

        # create masks
        files_mask = glob.glob(self.source_dir[1] + '*' + self.file_ext)
        print('%s files found' % len(files_mask))
        for file_i in files_mask:
            mask_i = tifffile.imread(file_i)
            mask_i[mask_i < 255] = 0
            # mask_i = 255 - mask_i
            mask_i = morphology.erosion(mask_i, morphology.octagon(self.dilate_mask,1))
            mask_i = mask_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            tifffile.imsave(self.mask_path + save_i + '.tif', mask_i.astype('int8'))

    def merge_images(self):
        self.mask_files = glob.glob(self.data_path + '/mask/*.tif')
        self.image_files = glob.glob(self.data_path + '/image/*.tif')

        if len(self.mask_files) != len(self.image_files):
            raise ValueError('Number of ground truth does not match number of image stacks')

        for i, file_i in enumerate(self.mask_files):
            basename_i = os.path.basename(file_i)
            mask_i = tifffile.imread(self.data_path + '/mask/' + basename_i)
            image_i = tifffile.imread(self.data_path + '/image/' + basename_i)
            # permute axis (1->3)
            # image_i = np.rollaxis(image_i,0,3)
            merge = np.zeros((mask_i.shape[0], mask_i.shape[1], 3))
            merge[:, :, 0] = mask_i
            merge[:, :, 1] = image_i
            merge = merge.astype('uint8')
            tifffile.imsave(self.merge_path + str(i) + '.tif', merge)

    def split(self):
        self.merges = glob.glob(self.merge_path + '*.tif')
        for i in range(len(self.merges)):
            merge = tifffile.imread(self.merge_path + str(i) + '.tif')
            dim_in = merge.shape
            # padding if dim_in < dim_out
            x_gap = max(0, self.dim_out[0] - dim_in[0])
            y_gap = max(0, self.dim_out[1] - dim_in[1])
            merge = np.pad(merge, ((0, x_gap), (0, y_gap), (0, 0)), 'reflect')
            # number of patches in x and y
            dim_in = merge.shape
            N_x = int(np.ceil(dim_in[0] / self.dim_out[0]))
            N_y = int(np.ceil(dim_in[1] / self.dim_out[1]))
            # starting indices of patches
            X_start = np.linspace(0, dim_in[0] - self.dim_out[0], N_x).astype('int16')
            Y_start = np.linspace(0, dim_in[1] - self.dim_out[1], N_y).astype('int16')
            for j in range(N_x):
                for k in range(N_y):
                    patch_ij = merge[X_start[j]:X_start[j] + self.dim_out[0], Y_start[k]:Y_start[k] + self.dim_out[1],
                               :]
                    tifffile.imsave(self.split_merge_path + '%s_%s_%s.tif' % (i, j, k), patch_ij)
                    tifffile.imsave(self.split_image_path + '%s_%s_%s.tif' % (i, j, k), patch_ij[:, :, 1])
                    tifffile.imsave(self.split_mask_path + '%s_%s_%s.tif' % (i, j, k), patch_ij[:, :, 0])

    def augment(self, p=0.8):
        aug_pipeline = Compose([
            Flip(),
            ShiftScaleRotate(0.05, 0.0, 90),
            GaussNoise(var_limit=(20, 50)),
            GaussianBlur(blur_limit=3),
            RandomBrightnessContrast(),
        ],
            p=p)

        patches_image = glob.glob(self.split_image_path + '*.tif')
        patches_mask = glob.glob(self.split_mask_path + '*.tif')
        n_patches = len(patches_image)
        k = 0
        for i in range(n_patches):
            image_i = tifffile.imread(patches_image[i])
            mask_i = tifffile.imread(patches_mask[i])

            data_i = {'image': image_i, 'mask': mask_i}
            data_aug_i = np.asarray([aug_pipeline(**data_i) for _ in range(self.aug_factor)])
            imgs_aug_i = np.asarray([data_aug_i[j]['image'] for j in range(self.aug_factor)])
            masks_aug_i = np.asarray([data_aug_i[j]['mask'] for j in range(self.aug_factor)])

            for j in range(self.aug_factor):
                tifffile.imsave(self.aug_image_path + '%s.tif' % (k), imgs_aug_i[j])
                tifffile.imsave(self.aug_mask_path + '%s.tif' % (k), masks_aug_i[j])
                k += 1
        print('Number of training images: %s' % k)

    def __len__(self):
        return len(os.listdir(self.aug_image_path))

    def __getitem__(self, idx):
        imgname = str(idx) + '.tif'
        midname = os.path.basename(imgname)
        image_0 = tifffile.imread(self.aug_image_path + midname).astype('float32') / 255
        mask_0 = tifffile.imread(self.aug_mask_path + midname).astype('float32') / 255
        image = torch.from_numpy(image_0)
        mask = torch.from_numpy(mask_0)
        del image_0, mask_0
        sample = {'image': image, 'mask': mask}
        return sample


# simple 2D conv net

class Unet(nn.Module):
    def __init__(self, n_filter=64, kernel_size=3):
        super().__init__()

        # encode
        self.encode1 = self.conv(1, n_filter)
        self.encode2 = self.conv(n_filter, n_filter)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode3 = self.conv(n_filter, 2 * n_filter)
        self.encode4 = self.conv(2 * n_filter, 2 * n_filter)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode5 = self.conv(2 * n_filter, 4 * n_filter)
        self.encode6 = self.conv(4 * n_filter, 4 * n_filter)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode7 = self.conv(4 * n_filter, 8 * n_filter)
        self.encode8 = self.conv(8 * n_filter, 8 * n_filter)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # middle
        self.middle_conv1 = self.conv(8 * n_filter, 16 * n_filter)
        self.middle_conv2 = self.conv(16 * n_filter, 16 * n_filter)

        # decode
        self.up1 = nn.ConvTranspose2d(16 * n_filter, 8 * n_filter, kernel_size=2, stride=2)
        self.decode1 = self.conv(16 * n_filter, 8 * n_filter)
        self.decode2 = self.conv(8 * n_filter, 8 * n_filter)
        self.up2 = nn.ConvTranspose2d(8 * n_filter, 4 * n_filter, kernel_size=2, stride=2)
        self.decode3 = self.conv(8 * n_filter, 4 * n_filter)
        self.decode4 = self.conv(4 * n_filter, 4 * n_filter)
        self.up3 = nn.ConvTranspose2d(4 * n_filter, 2 * n_filter, kernel_size=2, stride=2)
        self.decode5 = self.conv(4 * n_filter, 2 * n_filter)
        self.decode6 = self.conv(2 * n_filter, 2 * n_filter)
        self.up4 = nn.ConvTranspose2d(2 * n_filter, 1 * n_filter, kernel_size=2, stride=2)
        self.decode7 = self.conv(2 * n_filter, 1 * n_filter)
        self.decode8 = self.conv(1 * n_filter, 1 * n_filter)
        self.final = nn.Sequential(
            nn.Conv2d(n_filter, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def conv(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return block

    def concat(self, x1, x2):
        if x1.shape == x2.shape:
            return torch.cat((x1, x2), 1)
        else:
            print(x1.shape, x2.shape)
            raise ValueError('concatenation failed: wrong dimensions')

    def forward(self, x):
        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        m1 = self.maxpool1(e2)
        e3 = self.encode3(m1)
        e4 = self.encode4(e3)
        m2 = self.maxpool2(e4)
        e5 = self.encode5(m2)
        e6 = self.encode6(e5)
        m3 = self.maxpool2(e6)
        e7 = self.encode7(m3)
        e8 = self.encode8(e7)
        m4 = self.maxpool2(e8)

        mid1 = self.middle_conv1(m4)
        mid2 = self.middle_conv2(mid1)

        u1 = self.up1(mid2)
        c1 = self.concat(u1, e7)
        d1 = self.decode1(c1)
        d2 = self.decode2(d1)
        u2 = self.up2(d2)
        c2 = self.concat(u2, e5)
        d3 = self.decode3(c2)
        d4 = self.decode4(d3)
        u3 = self.up3(d4)
        c3 = self.concat(u3, e3)
        d5 = self.decode5(c3)
        d6 = self.decode6(d5)
        u4 = self.up4(d6)
        c4 = self.concat(u4, e1)
        d7 = self.decode7(c4)
        d8 = self.decode8(d7)
        out = self.final(d8)

        return out


class Trainer:
    def __init__(self, dataset, num_epochs, batch_size=4, lr=1e-3, momentum=1e-1, n_filter=64, val_split=0.1,
                 save_dir='./', save_name='model.pth',
                 load_weights=False):
        self.model = Unet(n_filter=n_filter).to(device)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.best_loss = torch.tensor(float('inf'))
        # split training and validation data
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        train_data, val_data = random_split(dataset, [num_train, num_val])
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.save_dir = save_dir
        self.save_name = save_name
        if load_weights:
            self.state = torch.load(self.save_dir + '/' + self.save_name)
            self.model.load_state_dict(self.state['state_dict'])
            self.best_loss = self.state['best_loss']

    def iterate(self, epoch, mode):
        loss_list = []
        if mode == 'train':
            print('Starting training epoch %s ...' % epoch)
            for i, batch_i in enumerate(Bar(self.train_loader)):
                x_i = batch_i['image'].view(self.batch_size, 1, 256, 256).to(device)
                y_i = batch_i['mask'].view(self.batch_size, 1, 256, 256).to(device)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.model(x_i)

                # Compute and print loss
                loss = self.criterion(y_pred, y_i)
                loss_list.append(loss)

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        elif mode == 'val':
            loss_list = []
            print('Starting validation epoch %s ...' % epoch)
            with torch.no_grad():
                for i, batch_i in enumerate(Bar(self.val_loader)):
                    x_i = batch_i['image'].view(self.batch_size, 1, 256, 256).to(device)
                    y_i = batch_i['mask'].view(self.batch_size, 1, 256, 256).to(device)
                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred = self.model(x_i)
                    loss = self.criterion(y_pred, y_i)
                    loss_list.append(loss)
            val_loss = torch.stack(loss_list).mean()
            return val_loss

        torch.cuda.empty_cache()

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, 'train')
            self.state = {
                'epoch': epoch,
                'best_loss': self.best_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, 'val')
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print('Validation loss improved from %s to %s - saving model state' % (
                    round(self.best_loss.item(), 5), round(val_loss.item(), 5)))
                self.state['best_loss'] = self.best_loss = val_loss
                torch.save(self.state, self.save_dir + '/' + self.save_name)


class Predict:
    '''
    Class for prediction of tif-movies.
    1) Loading file and preprocess (normalization)
    2) Resizing of images into patches with resize_dim
    3) Prediction with U-Net
    4) Stitching of predicted patches and averaging of overlapping regions
    '''

    def __init__(self, tif_file, result_name, model_params, n_filter=64, resize_dim=(512, 512), invert=False,
                 frame_lim=None, clip_thrs=99.8):
        self.tif_file = tif_file
        self.resize_dim = resize_dim
        self.n_filter = n_filter
        self.invert = invert
        self.clip_thrs = clip_thrs
        self.frame_lim = frame_lim
        self.result_name = result_name
        if self.result_name == 'nodes':
            self.folder = os.path.dirname(self.tif_file)
        else:
            self.folder = re.split('.tif', self.tif_file)[0] + '/'
        # os.makedirs(self.folder, exist_ok=True)
        # self.npy_path = self.folder + '/data/'
        # os.makedirs(self.npy_path, exist_ok=True)
        #         self.get_and_save_metadata(tif_file)

        # read, preprocess and split data
        self.read_data()
        self.preprocess()
        self.split()

        # load model and predict data
        self.model = Unet(n_filter=self.n_filter).to(device)
        self.model.load_state_dict(torch.load(model_params)['state_dict'])
        self.model.eval()
        self.predict()

        # stitch patches (mean of overlapped regions)
        self.stitch()

        # save as .tif file
        self.save_as_tif(self.imgs_result, self.result_name)

    def open_folder(self):
        if platform.system() == "Windows":
            os.startfile(self.folder)
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", self.folder])

    def read_data(self, plot_first=True):
        self.imgs = tifffile.imread(self.tif_file)
        if self.frame_lim != None:
            self.imgs = self.imgs[self.frame_lim[0]:self.frame_lim[1]]
        self.imgs_shape = self.imgs.shape
        if len(self.imgs_shape) == 2:  # if single image
            self.imgs_shape = [1, self.imgs_shape[0], self.imgs_shape[1]]
        if plot_first:
            plt.figure(figsize=(16, 6))
            try:
                plt.imshow(self.imgs[0], cmap='Greys')
            except:
                plt.imshow(self.imgs, cmap='Greys')
            plt.show()

    def preprocess(self):
        self.imgs = np.clip(self.imgs, np.percentile(self.imgs, 0.2), np.percentile(self.imgs, self.clip_thrs))
        for i, img in enumerate(self.imgs):
            img = img - np.min(img)
            img = img / np.max(img)
            img = img.astype('float32')
            if self.invert:
                img = 1 - img
            self.imgs[i] = img

    def split(self):
        # number of patches in x and y
        self.N_x = int(np.ceil(self.imgs_shape[1] / self.resize_dim[0]))
        self.N_y = int(np.ceil(self.imgs_shape[2] / self.resize_dim[1]))
        self.N_per_img = self.N_x * self.N_y
        self.N = self.N_x * self.N_y * self.imgs_shape[0]  # total number of patches
        print('Resizing into each %s patches ...' % self.N_per_img)

        # define array for prediction
        self.patches = np.zeros((self.N, 1, self.resize_dim[0], self.resize_dim[1]), dtype=np.float32)

        # zero padding of image if imgs_shape < resize_dim
        if self.imgs_shape[0] > 1:
            if self.imgs_shape[1] < self.resize_dim[0]:  # for x
                self.imgs = np.pad(self.imgs, ((0, 0), (0, self.resize_dim[0] - self.imgs_shape[1]), (0, 0)),
                                   'constant')
            if self.imgs_shape[2] < self.resize_dim[1]:  # for y
                self.imgs = np.pad(self.imgs, ((0, 0), (0, 0), (0, self.resize_dim[1] - self.imgs_shape[2])),
                                   'constant')
        elif self.imgs_shape[0] == 1:
            if self.imgs_shape[1] < self.resize_dim[0]:  # for x
                self.imgs = np.pad(self.imgs, ((0, self.resize_dim[0] - self.imgs_shape[1]), (0, 0)), 'constant')
            if self.imgs_shape[2] < self.resize_dim[1]:  # for y
                self.imgs = np.pad(self.imgs, ((0, 0), (0, self.resize_dim[1] - self.imgs_shape[2])), 'constant')

        # starting indices of patches
        self.X_start = np.linspace(0, self.imgs_shape[1] - self.resize_dim[0], self.N_x).astype('int16')
        self.Y_start = np.linspace(0, self.imgs_shape[2] - self.resize_dim[1], self.N_y).astype('int16')

        # split in resize_dim
        n = 0
        if self.imgs_shape[0] > 1:
            for i, img in enumerate(self.imgs):
                for j in range(self.N_x):
                    for k in range(self.N_y):
                        self.patches[n, 0, :, :] = img[self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                                   self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                        n += 1
        elif self.imgs_shape[0] == 1:
            for j in range(self.N_x):
                for k in range(self.N_y):
                    self.patches[n, 0, :, :] = self.imgs[self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                               self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                    n += 1

    def predict(self):
        self.results = np.zeros(self.patches.shape)
        print('Predicting data ...')
        with torch.no_grad():
            for i, patch_i in enumerate(tqdm(self.patches)):
                patch_i = torch.from_numpy(patch_i).to(device)
                patch_i = patch_i.view((1, 1, self.resize_dim[0], self.resize_dim[1]))
                res_i = self.model(patch_i).view((1, self.resize_dim[0], self.resize_dim[1]))
                res_i = res_i.cpu().numpy()
                self.results[i] = res_i

    def stitch(self):
        print('Stitching patches back together ...')
        # create array
        self.imgs_result = np.zeros(self.imgs.shape)

        for i in range(self.imgs_shape[0]):
            if self.imgs_shape[0] > 1:  # if stack
                stack_result_i = np.zeros((self.N_per_img, self.imgs.shape[1], self.imgs.shape[2])) * np.nan
            elif self.imgs_shape[0] == 1:  # if only one image
                stack_result_i = np.zeros((self.N_per_img, self.imgs.shape[0], self.imgs.shape[1])) * np.nan
            n = 0
            for j in range(self.N_x):
                for k in range(self.N_y):
                    stack_result_i[n, self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                    self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]] = self.results[i * self.N_per_img + n, 0, :,
                                                                            :]
                    # average overlapping regions
                    if self.imgs_shape[0] > 1:  # if stack
                        self.imgs_result[i] = np.nanmean(stack_result_i, axis=0)
                    elif self.imgs_shape[0] == 1:  # if only one image
                        self.imgs_result = np.nanmean(stack_result_i, axis=0)
                    n += 1

        # change to input size (if zero padding)
        if self.imgs_shape[0] > 1:  # if stack
            self.imgs_result = self.imgs_result[:, :self.imgs_shape[1], :self.imgs_shape[2]]
        elif self.imgs_shape[0] == 1:  # if only one image
            self.imgs_result = self.imgs_result[:self.imgs_shape[1], :self.imgs_shape[2]]

    def save_as_tif(self, imgs, filename):
        imgs = imgs - np.nanmin(imgs)
        imgs /= np.nanmax(imgs)
        imgs *= 255
        imgs = imgs.astype('uint8')
        if len(imgs.shape) == 3:  # if stack
            with tifffile.TiffWriter(filename) as tiff:
                for img in imgs:
                    tiff.save(img)
        elif len(imgs.shape) == 2:  # if single image
            tifffile.imsave(filename, imgs)
        print('Saving prediction results as %s' % filename)
