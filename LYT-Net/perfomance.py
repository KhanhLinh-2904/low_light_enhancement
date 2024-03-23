
import time
import torch
import scripts.data_loading as dl
import tensorflow as tf
from model.arch import LYT, Denoiser
import matplotlib.pyplot as plt
from tqdm import tqdm

def lowlight(img_path, model):
    img = dl.load_image_test(img_path,0)
    img = tf.expand_dims(img, axis=0)
    start = time.time()
    generated_image = model(img)
    end_time = (time.time() - start)
    print(end_time)
    return end_time
    
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
        
        filePath = '/home/linhhima/Downloads/Linh01.png'
        sum_time = 0
        for i in range(0, 110):
            inference_time = lowlight(filePath, model)
            if i < 10:
                continue
            sum_time = sum_time + inference_time
        print("Perfomance LYT: ", sum_time/100)