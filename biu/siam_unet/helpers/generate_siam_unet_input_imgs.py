import cv2
import numpy as np
import tifffile
import os
import glob

def generate_coupled_image(movie, frame, output):
    """
    Generates an image from the previous frame of the given frame and the given frame of the given movie. Should be used as input to Siam_UNet. The output image is a tiff file with the files like this, with the output image of twice the width of the input.
    [    Previous   |     Frame to be    ]
    [     Frame     |   Inferred From    ]

    Note on arg: 
        frame (int): 0-indexed
    """

    curr_frame = tifffile.imread(movie, key=frame)
    if frame == 0:
        prev_frame = tifffile.imread(movie, key=(frame + 1))
    else:
        prev_frame = tifffile.imread(movie, key=(frame - 1))

    if curr_frame is None:
        print(f'Error: {movie} not found.')
        raise IOError   # tiff file not found

    out = np.concatenate((prev_frame, curr_frame), axis=1).astype(np.uint8)
    cv2.imwrite(filename=output, img=out, )

def generate_coupled_image_from_self(img, output_img):
    """
        Generates an input image for siam by concatenating an image with itself 
    """

    curr_frame = tifffile.imread(img)

    if curr_frame is None:
        raise IOError   # tiff file not found

    out = np.concatenate((curr_frame, curr_frame), axis=1).astype(np.uint8)
    cv2.imwrite(filename=output, img=out, )

def utilize_search_result(search_result_mr_txt, movie_path_prefix, labels_path_prefix, output_folder):
    """Parses search results obtained from find_frame_of_image.find_frame_of_image()'s machine readable output and pass them to generate_coupled_image() to create training data for Siam_UNet. 

    This function outputs two folders under output_folder: image and label. All the output will have a suffix of .tif, and thyey will have the same names as the names of input labels. The output in the image directory will be the current frame concatenated with the previous frame, while the label directory will contain the corresponding labels, directly copied from the input labels directory. Visually, the directory structure at output:

    output_folder
    |
    |----image
    |       |-----concatenated images
    |
    |
    |----label
    |       |-----labels copied from the input labels directory

    Args:
        search_result_mr_txt (str): path to the machine readable output
        movie_path_prefix, labels_path_prefix (str): prefixes to name of each movie in the search result file, respectively
        output_folder (str): path to the output folder. See above.
    """

    image_output_folder = output_folder + '/image/'
    label_output_folder = output_folder + '/label/'

    # make output folder if it does not exist
    os.system(f'mkdir -p \'{output_folder}\'')
    # recreate image and label output folders
    os.system(f'rm -rf \'{image_output_folder}\'')
    os.system(f'rm -rf \'{label_output_folder}\'')
    os.system(f'mkdir -p \'{image_output_folder}\'')
    os.system(f'mkdir -p \'{label_output_folder}\'')

    with open(search_result_mr_txt) as sr:
        lines = sr.readlines()
        for line in lines:
            label_path = labels_path_prefix + '/' + line.split('\t')[0]
            image_path = movie_path_prefix + '/' +  line.split('\t')[1]
            output_image_path = image_output_folder + line.split('\t')[0]
            frame_number = int(line.split('\t')[2])
            # copy label to label folder
            os.system(f'cp \'{label_path}\' \'{label_output_folder}\'')
            # create concatenated image in image_folder
            generate_coupled_image(image_path, frame=frame_number, output=output_image_path)
    
    for file in glob.glob(f'{label_output_folder}/*.tif'):
        i = cv2.imread(file)
        fo = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
        os.remove(file)
        cv2.imwrite(file, fo)



if __name__ == '__main__':
    import constants
    # utilize_search_result(f'{constants.RAZER_LOCAL_BASE_DIR}/training_data/training_data/yokogawa/lateral_epidermis/search_result_mr.txt', f'{constants.RAZER_LOCAL_ALL_MOVIES_DIR}', f'{constants.RAZER_LOCAL_BASE_DIR}/training_data/training_data/yokogawa/lateral_epidermis/label/', f'{constants.RAZER_LOCAL_BASE_DIR}/training_data/training_data/yokogawa/temp_test/')
    
    utilize_search_result(f'{constants.RAZER_LOCAL_BASE_DIR}/training_data/training_data/yokogawa/amnioserosa/search_result_mr.txt', f'{constants.RAZER_LOCAL_ALL_MOVIES_DIR}', f'{constants.RAZER_LOCAL_BASE_DIR}/training_data/training_data/yokogawa/amnioserosa/label/', f'{constants.RAZER_LOCAL_BASE_DIR}/training_data/training_data/yokogawa/siam_amnioserosa_sanitize_test/')
