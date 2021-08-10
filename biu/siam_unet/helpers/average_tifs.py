import cv2 # type: ignore
import numpy as np
from util import write_info_file

import platform
if platform.system() != 'Linux':
    raise Exception  # this script is designed to use Linux bash commands. Please use Linux

def average_tifs(inputs : list, output):
	tifs = []
	for tiff_movie in inputs:
		retval, imgs = cv2.imreadmulti(tiff_movie)
		if not retval:
			print(f'Error: {tiff_movie} not found.')
			raise IOError   # tiff file not found
		tifs.append(imgs)
	print('Averaging', len(tifs), 'images')
	out_tifs = [] # list (frames) of numpy arrays that are of size e.g. 1160*2300
	for i in range(len(tifs[0])):
		out_tifs.append(np.zeros(tifs[0][0].shape, dtype=np.int16))

	# add each frame from the input to the output
	for tif_idx, tif in enumerate(tifs):
		for frame_idx, frame in enumerate(tif):
			# add each frame to the corresponding frame in out_tif
			out_tifs[frame_idx] += frame

	write_info_file(f'{output}.info.txt', f'Operation: Average\nOutfile: {output}\nInput files:{inputs}')
	final_output_tif = []
	# average the frames in the output
	for frame in out_tifs:
		final_output_tif.append((frame / len(tifs)).astype('uint8'))
	
	cv2.imwritemulti(output, final_output_tif)


if __name__ == '__main__':
	# average_tifs(['/home/longyuxi/Documents/mount/predicted_out/predicted_newly_drawn_21B11-shgGFP-kin-18-bro4_epoch_500.tif', '/home/longyuxi/Documents/mount/predicted_out/predicted_newly_drawn_21B11-shgGFP-kin-18-bro4_epoch_500.tif'], 'z.tif')
	# average_tifs(['/home/longyuxi/Documents/mount/predicted_out/predicted_newly_drawn_21C04_shgGFP_kin_2_Pos4_epoch_500.tif', '/home/longyuxi/Documents/mount/predicted_out/old_self_model_predictions/predicted_old_model_21C04_shgGFP_kin_2_Pos4.tif'], '/home/longyuxi/Documents/mount/averaged_tifs/21C04_averaged.tif')
	from tif_to_mp4 import convert_to_mp4
	# for threshold_val in [50, 100, 150, 200, 250]:
	# 	convert_to_mp4('/home/longyuxi/Documents/mount/averaged_tifs/21C04_averaged.tif', f'/home/longyuxi/Documents/mount/averaged_tifs/21C04_averaged_thresholded_threshold_{threshold_val}.mp4', perform_threshold=True, threshold_val=threshold_val)
	convert_to_mp4('/home/longyuxi/Documents/mount/averaged_tifs/21C04_averaged.tif', '/home/longyuxi/Documents/mount/averaged_tifs/21C04_averaged.mp4')