import os
import shutil
import time
import gc

import numpy as np
import tifffile

import bio_image_unet.siam_unet as siam
import bio_image_unet.unet as unet
from bio_image_unet import unet3d
# create test folder with random training and test data
from bio_image_unet.progress import ProgressNotifier

folder = './temp_test/'


def test_unet():
    folder_image = folder + 'training_data/image/'
    folder_mask = folder + 'training_data/mask/'
    folder_results = folder + 'results/'
    os.makedirs(folder_image, exist_ok=True)
    os.makedirs(folder_mask, exist_ok=True)
    os.makedirs(folder_results, exist_ok=True)

    for i in range(5):
        # regular unet
        random_image = np.random.randint(0, 255, (128, 128))
        random_mask = np.random.randint(0, 255, (128, 128))
        tifffile.imwrite(folder_image + f'{i}.tif', random_image)
        tifffile.imwrite(folder_mask + f'{i}.tif', random_mask)

    random_movie = np.random.randint(0, 255, (20, 128, 128))
    tifffile.imwrite(folder + 'movie.tif', random_movie)

    # create training data set
    data = unet.DataProcess(source_dir=(folder_image, folder_mask), dim_out=(64, 64), data_path=folder + 'data/')

    # train
    train = unet.Trainer(data, num_epochs=4, n_filter=8, save_dir=folder + 'models_unet/')
    train.start()

    # predict movie
    unet.Predict(folder + 'movie.tif', result_name=folder_results + 'movie.tif',
                 model_params=folder + 'models_unet/model.pth', resize_dim=(64, 64),
                 progress_notifier=ProgressNotifier())


def test_siam_unet():
    folder_image_siam = folder + 'training_data_siam/image/'
    folder_mask_siam = folder + 'training_data_siam/mask/'
    folder_data = folder + 'data/'
    folder_results = folder + 'results/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder_image_siam, exist_ok=True)
    os.makedirs(folder_mask_siam, exist_ok=True)
    os.makedirs(folder_data, exist_ok=True)
    os.makedirs(folder_results, exist_ok=True)

    for i in range(5):
        # siam unet
        random_image = np.random.randint(0, 255, (2, 128, 128))
        random_mask = np.random.randint(0, 255, (128, 128))
        tifffile.imwrite(folder_image_siam + f'{i}.tif', random_image)
        tifffile.imwrite(folder_mask_siam + f'{i}.tif', random_mask)

    random_movie = np.random.randint(0, 255, (20, 128, 128))
    tifffile.imwrite(folder + 'movie.tif', random_movie)

    # create training data set
    data_siam = siam.DataProcess(source_dir=(folder_image_siam, folder_mask_siam), dim_out=(64, 64),
                                 data_path=folder + 'data_siam/')

    # train
    train_siam = siam.Trainer(data_siam, num_epochs=4, n_filter=8, save_dir=folder + 'models_siam/', load_weights=None)
    train_siam.start()

    # predict movie
    siam.Predict(folder + 'movie.tif', result_name=folder_results + 'movie.tif',
                 model_params=folder + 'models_siam/model.pth', resize_dim=(64, 64),
                 progress_notifier=ProgressNotifier())

def test_unet3d():
    folder_image = folder + 'training_data_3d/image/'
    folder_mask = folder + 'training_data_3d/mask/'
    folder_results = folder + 'results/'
    os.makedirs(folder_image, exist_ok=True)
    os.makedirs(folder_mask, exist_ok=True)
    os.makedirs(folder_results, exist_ok=True)

    for i in range(5):
        # regular unet
        random_image = np.random.randint(0, 255, (32, 128, 128))
        random_mask = np.random.randint(0, 255, (32, 128, 128))
        tifffile.imwrite(folder_image + f'{i}.tif', random_image)
        tifffile.imwrite(folder_mask + f'{i}.tif', random_mask)

    random_movie = np.random.randint(0, 255, (32, 128, 128))
    tifffile.imwrite(folder + 'movie.tif', random_movie)

    # create training data set
    data = unet3d.DataProcess(source_dir=(folder_image, folder_mask), dim_out=(32, 64, 64), data_path=folder + 'data/')

    # train
    train = unet3d.Trainer(data, num_epochs=4, n_filter=8, save_dir=folder + 'models_unet3d/')
    train.start()

    # predict movie
    unet3d.Predict(folder + 'movie.tif', result_name=folder_results + 'movie.tif',
                 model_params=folder + 'models_unet3d/model.pth', resize_dim=(16, 64, 64),
                 progress_notifier=ProgressNotifier())


def delete_folder_with_retry(folder, max_attempts=5, wait_seconds=2):
    for attempt in range(max_attempts):
        try:
            gc.collect()  # Force garbage collection
            shutil.rmtree(folder)
            print(f"Successfully deleted {folder}")
            break
        except PermissionError as e:
            print(f"PermissionError on attempt {attempt + 1}: {e}")
            time.sleep(wait_seconds)
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    else:
        print(f"Failed to delete {folder} after {max_attempts} attempts.")


if __name__ == "__main__":
    test_unet()
    test_siam_unet()
    test_unet3d()
    # delete test folder
    delete_folder_with_retry(folder)
    print("*" * 20 + " \nTests completed successfully")
