import numpy as np
from skimage import io
import os
import skimage
from skimage import data
from skimage import restoration
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import img_as_float
from skimage import filters
#from skimage import filter as filters

check = np.zeros((9,9))
check[::2, 1::2] = 1
check[1::2, ::2] = 1
plt.figure(1)
plt.imshow(check, cmap = 'gray', interpolation = 'nearest')

camera = data.camera()
camera_hist, camera_bins = np.histogram(camera, bins = 10)
camera_vector = camera.reshape(camera.shape[0]*camera.shape[1],1)
plt.figure('hist')
plt.hist(camera_vector, camera_bins)

plt.figure('2')
plt.imshow(camera)
io.imsave('camera.png', camera)

APP_path = os.path.dirname(os.path.abspath(__file__))
APP_base_path = os.path.basename(APP_path)
head_path, tail_path = os.path.split(APP_path)

filepath = os.path.join(APP_path, 'camera.png')

camera = data.camera()
camera_multiply = 3 * camera

camera_float = img_as_float(camera)
print ('camera.max(), camera_float.max(): ', camera.max(), camera_float.max())

# use io to read and save image
logo = io.imread('http://scikit-image.org/_static/img/logo.png')
io.imsave('local_logo.png', logo)

text = data.text()
hsobel_text = filters.hsobel(text)

camera = data.camera()

camera_equalized = exposure.equalize_hist(camera)

plt.show()
