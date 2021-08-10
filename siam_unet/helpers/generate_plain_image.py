import cv2
import os
os.nice(20)
import subprocess
import numpy as np

def generate_plain_image(pixel_value=255, shape=(1000, 500), outfile_name="val_255.png"):
	out = (np.ones(shape) * pixel_value).astype(np.uint8)
	cv2.imwrite(filename=outfile_name, img=out, )

if __name__ == '__main__':
	generate_plain_image()