'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2023-02-24 14:25:00
'''
import cv2, os
import numpy as np
import BM3D
import KSVD
from PIL import Image
from time import time
# from Evaluation import *


# Non-Local Means
def NLM(img_path):
    img_noise = cv2.imread(img_path, 0)
    img_noise = np.clip(np.round(img_noise * 1.0), 0, 255).astype(np.uint8)
    img = cv2.fastNlMeansDenoising(img_noise, None, 15.0, 7, 11)
    return img

# BM3D
def BM3D_Use(img_path, img_size):
    Basic_img, Final_img = BM3D.BM3D(img_path, img_size)
    return Basic_img, Final_img

# K-SVD
def K_SVD(img_path):
    im_ascent = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ksvd = KSVD.KSVD(300)
    dictionary, sparsecode = ksvd.fit(im_ascent)
    return dictionary.dot(sparsecode)


if __name__ == "__main__":
    # 1_NLM 2_BM3D 3_KSVD 
    # path = "./Data/"
    # destpath = "./Results/1_NLM/"
    # file_list = os.listdir(path)
    img_path = "/Users/adam_lau/MIPAV/DeepLearning/denoise_traditional/image_base.png"
    times = []
    # for i in file_list:
    # img_path = "{}{}".format(path, i)
    # dest_img = "{}{}".format(destpath, i)
    # print(i)

    # NLM
    start = time()

    NLM_img = NLM(img_path)
    dest_img = "/Users/adam_lau/MIPAV/DeepLearning/denoise_traditional/image_base_nlm.png"
    cv2.imwrite(dest_img, NLM_img)
    # img.save(dest_img)
    
    end = time()
    cost_time = end - start
    print(cost_time)

    # BM3D
    start = time()
    
    Basic_img, Final_img = BM3D_Use(img_path, 512)
    dest_img = "/Users/adam_lau/MIPAV/DeepLearning/denoise_traditional/image_base_bm3d.png"
    cv2.imwrite(dest_img, Final_img)
    # Final_img = cv2.imread(dest_img).astype('float32') 
    # Final_img = cv.resize(Final_img, dsize=(NLM_img.shape[1], NLM_img.shape[0]))
    # cv2.imwrite(dest_img, Final_img)

    end = time()
    cost_time = end - start
    print(cost_time)

    # K-SVD
    start = time()

    KSVD_img = K_SVD(img_path)
    dest_img = "/Users/adam_lau/MIPAV/DeepLearning/denoise_traditional/image_base_kvsd.png"
    cv2.imwrite(dest_img, KSVD_img)

    end = time()
    cost_time = end - start
    print(cost_time)

        # if i != "1.png":
        #     times.append(cost_time)
        
  