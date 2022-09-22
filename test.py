import os
import shutil
import numpy as np
import tifffile

import biu.unet as unet
import biu.siam_unet as siam

# create test folder with random training and test data

folder = './temp_test/'
folder_image = folder + 'training_data/image/'
folder_mask = folder + 'training_data/mask/'
folder_image_siam = folder + 'training_data_siam/image/'
folder_mask_siam = folder + 'training_data_siam/mask/'
folder_data = folder + 'data/'
folder_results = folder + 'results/'
os.makedirs(folder, exist_ok=True)
os.makedirs(folder_mask, exist_ok=True)
os.makedirs(folder_image, exist_ok=True)
os.makedirs(folder_image_siam, exist_ok=True)
os.makedirs(folder_mask_siam, exist_ok=True)
os.makedirs(folder_data, exist_ok=True)
os.makedirs(folder_results, exist_ok=True)

for i in range(5):
    # regular unet
    random_image = np.random.randint(0, 255, (128, 128))
    random_mask = np.random.randint(0, 255, (128, 128))
    tifffile.imwrite(folder_image + f'{i}.tif', random_image)
    tifffile.imwrite(folder_mask + f'{i}.tif', random_mask)
    # siam unet
    random_image = np.random.randint(0, 255, (2, 128, 128))
    random_mask = np.random.randint(0, 255, (128, 128))
    tifffile.imwrite(folder_image_siam + f'{i}.tif', random_image)
    tifffile.imwrite(folder_mask_siam + f'{i}.tif', random_mask)


random_movie = np.random.randint(0, 255, (20, 128, 128))
tifffile.imwrite(folder + 'movie.tif', random_movie)

# create training data set
data = unet.DataProcess(source_dir=(folder_image, folder_mask), dim_out=(64, 64), data_path=folder+'data/')
data_siam = siam.DataProcess(source_dir=(folder_image_siam, folder_mask_siam), dim_out=(64, 64),
                             data_path=folder+'data_siam/')

# train
train = unet.Trainer(data, num_epochs=4, n_filter=8, save_dir=folder + 'models_unet/')
train.start()
train_siam = siam.Trainer(data_siam, num_epochs=4, n_filter=8, save_dir=folder + 'models_siam/', load_weights=None)
train_siam.start()

# predict movie
unet.Predict(folder + 'movie.tif', result_name=folder_results + 'movie.tif',
             model_params=folder + 'models_unet/model.pth', resize_dim=(64, 64))
siam.Predict(folder + 'movie.tif', result_name=folder_results + 'movie.tif',
             model_params=folder + 'models_siam/model.pth', resize_dim=(64, 64))

# delete test folder
# shutil.rmtree(folder)


