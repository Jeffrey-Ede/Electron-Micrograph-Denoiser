import numpy as np
from denoiser import Denoiser, disp

#Create a 1500x1500 image from random numbers for demonstration
#Try replacing this with your own image!
img = np.random.rand(1500, 1500)

#Replace with the location of your saved checkpoint
checkpoint_loc = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13/model"

#Initialize the denoising neural network
noise_remover = Denoiser(checkpoint_loc=checkpoint_loc, visible_cuda="0")

#Denoise a 512x512 crop from the image
crop = img[:512,:512]
denoised_crop = noise_remover.denoise_crop(crop)

#Denoise the entire image
denoised_img = noise_remover.denoise(img)

disp(crop) #Crop before denoising
disp(denoised_crop) #Crop after denoising
disp(img) #Image before denoising
disp(denoised_img) #Image after denoising
