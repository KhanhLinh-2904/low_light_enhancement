
import glob
import os
import time
import numpy as np
import torch
import scripts.data_loading as dl
import tensorflow as tf
from model.arch import LYT, Denoiser
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


def save_images(tensor, save_path, image_name):
    image_numpy = np.round(tensor[0].numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, image_name), cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB))

def lowlight(img_path, model):
    img = dl.load_image_test(img_path,0)
    print("shape of image : ", img.shape)

    img = tf.expand_dims(img, axis=0)
    print("-------------------- shape image: ", img.shape)
    start = time.time()
    generated_image = model(img)
    end_time = (time.time() - start)
    print(end_time)
    return generated_image
    
if __name__ == '__main__':
   # Build model
    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None,None,None,1))
    denoiser_cr.build(input_shape=(None,None,None,1))
    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None,None,None,3))

    # Loading weights
    weights_path = 'pretrained_weights/LOL_v1_weights.h5'
    model.load_weights(f'{weights_path}')

    with torch.no_grad():
        
        filePath = '/home/linhhima/low_light_enhancement/LYT-Net/data/test_image/Real'
        save_path = '/home/linhhima/low_light_enhancement/LYT-Net/results/test_image/Real'
        file_list = os.listdir(filePath)
        sum_time = 0
        for file_name in file_list:
            path_to_image = os.path.join(filePath, file_name)
            print("path_to_image:",path_to_image)
            inference_image = lowlight(path_to_image, model)
            inference_image = (inference_image + 1.0) / 2.0
            save_images(inference_image, save_path, file_name)
        print("DOne")