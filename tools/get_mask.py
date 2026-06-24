import imageio
import os
import numpy as np
import tqdm
import cv2

image_dir = '/home/home/ly/EventDyN/data/umbrella/dense/motion_masks'
img_names = np.array(sorted(os.listdir(image_dir)))  # all image names

img_paths = [os.path.join(image_dir, n) for n in img_names]

#kernel = np.ones((10, 10), np.uint8)
for p in img_paths:
    pp = p.replace('_pseudo.png','.png')
    """img = imageio.imread(p)
    erosion = cv2.erode(img, kernel)
    imageio.imsave(pp,erosion) """
    img = imageio.imread(p)[:, :, 0]#.astype(np.int32)  # (H, W, 3) np.uint8
    # img = np.stack([img[...,2], img[...,1], img[...,0]], -1)
    #img = ((img-255) * -1).astype(np.uint8)
    imageio.imsave(pp,img)