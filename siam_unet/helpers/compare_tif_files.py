from find_frame_of_image import mse
import numpy as np
import tifffile

a = tifffile.imread('/media/longyuxi/H is for HUGE/docmount backup/unet_pytorch/data/mask/3.tif')

b = tifffile.imread('/media/longyuxi/H is for HUGE/docmount backup/unet_pytorch/training_data/training_data/yokogawa/siam_amnioserosa/label/3.tif')

# c = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('3.tif', c)
# d = cv2.imread('3.tif')
# print(d.shape)

a = np.array(a)
b = np.array(b)

print(a.shape, b.shape)
print(mse(a, b))
print(np.average(a), np.average(b))
