import os

import cv2
import numpy as np
import torch
from lpips import LPIPS
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import compare_ssim
from math import log10, sqrt
from scipy import signal
from PIL import Image 

def check_empty_img(img): 
    # Reading Image 
    # You can give path to the  
    # image as first argument 
    image = cv2.imread(img) 
  
    # Checking if the image is empty or not 
    if image is None: 
        result = "Image is empty!!"
    else: 
        result = "Image is not empty!!"
  
    return result 
      

def cal_ssim(img1, img2):
    
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T
     
    M,N = np.shape(img1)

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    img1 = np.float64(img1)
    img2 = np.float64(img2)
 
    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim,ssim_map

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE. The lower the error, the more "similar" the two images are.
    return mse_error

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def calculate_metrics(original_img, enhanced_img):
    # Chuyển đổi ảnh từ [0, 255] sang [0, 1]
    # print(type(original_img))
    # original_img = np.array(original_img)
    # enhanced_img = np.array(original_img)

    original_img = original_img.astype(float) / 255.0
    enhanced_img = enhanced_img.astype(float) / 255.0
    # original_img.resize(enhanced_img.shape)

    print('enhanced_img.shape: ', enhanced_img.shape)
    print('original_img.shape: ', original_img.shape)
    # original_img = original_img.resize(enhanced_img.shape)
    original_img = cv2.resize(original_img, (enhanced_img.shape[1], enhanced_img.shape[0]))

    # PSNR
    # print(original_img)
    psnr_value = psnr(original_img, enhanced_img, data_range=1)
    # psnr_value = PSNR(original_img, enhanced_img)
    # print(psnr_value)

    # SSIM
    ssim_value = ssim(original_img, enhanced_img, data_range=1)
    # (ssim_value, diff) = cal_ssim(original_img, enhanced_img)
    # Entropy
    entropy_original = entropy(original_img.flatten())
    entropy_enhanced = entropy(enhanced_img.flatten())

    # Standard Deviation
    std_dev_original = np.std(original_img)
    std_dev_enhanced = np.std(enhanced_img)
    original_tensor = torch.from_numpy(original_img)
    enhanced_tensor = torch.from_numpy(enhanced_img)

    # lpips_value = lpips_model(original_tensor, enhanced_tensor).item()
    mse_value = mse(original_img, enhanced_img)
    return (
        psnr_value,
        ssim_value,
        entropy_original,
        entropy_enhanced,
        std_dev_original,
        std_dev_enhanced,
        mse_value,
    )


if __name__ == '__main__':
    # lpips_model = LPIPS(net='vgg')
    original_image_folder = '/home/user/low_light_enhancement/Zero-DCE++/data/label_Test_Part2'
    enhanced_image_folder = '/home/user/low_light_enhancement/Zero-DCE++/data/result_Test_Part2_pretrained'

    images = os.listdir(enhanced_image_folder)
    results = ''
    for image in images:
        # print("image: ",image)
        enhanced_image_path = os.path.join(enhanced_image_folder, image)
        label_image_name = image.split('_')[0] + '.jpg'
        # print("label_image_name: ",label_image_name)
        original_image_path = os.path.join(original_image_folder, label_image_name)
        original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        enhanced_img = cv2.imread(enhanced_image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None or enhanced_img is None:
            print("enhanced_img path None:", enhanced_image_path)
            print("original_img path None:", original_image_path)
            continue
        if original_img is not None:
            (
                psnr_value,
                ssim_value,
                entropy_original,
                entropy_enhanced,
                std_dev_original,
                std_dev_enhanced,
                mse_value,
            ) = calculate_metrics(original_img, enhanced_img)
            result = f'{image}\npsnr: {psnr_value}, ssim: {ssim_value}, entropy: {entropy_enhanced}, std: {std_dev_enhanced}, mse: {mse_value}'
            results = results + result + '\n'
    print(results)
