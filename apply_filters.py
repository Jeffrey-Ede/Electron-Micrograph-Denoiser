import numpy as np
import cv2

import os
from random import shuffle

from scipy.misc import imread
from PIL import Image

from skimage.restoration import unsupervised_wiener, denoise_wavelet, \
    denoise_tv_chambolle, denoise_nl_means, denoise_tv_bregman
from scipy.signal import wiener
from skimage.measure import compare_ssim as ssim

from denoiser import Denoiser

#img1 = imread(r'Z:\Jeffrey-Ede\models\denoiser-multi-gpu-10\output-1.tif', mode='F')
#img2 = imread(r'Z:\Jeffrey-Ede\models\denoiser-multi-gpu-10\truth-1.tif', mode='F')
#print(ssim(img1, img2, data_range=img2.max()-img2.min()))
#print(np.mean((img1-img2)**2))

import time

in_dir = "E:/stills_hq-mini/"
save_dir = "E:/stills_hq-mini/denoiser-13-nn-extra-stats/"

subsets = ["train", "val", "test"]

def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def gen_lq(img, scale):
    '''Generate low quality image'''

    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the
    # correct average counts
    lq = np.random.poisson( img * scale )

    return scale0to1(lq)

def metric_no_metric(input, truth):

    input = input.astype(np.float32)
    mse = np.sum((input-truth)**2) / input.size
    struct_sim = ssim(input, truth, data_range=input.max()-input.min())

    return mse, struct_sim

def metric_gaussian(input, truth):

    filtered = cv2.GaussianBlur(input, (3,3), 0)
    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    return mse, struct_sim

def metric_bilateral(input, truth):

    filtered = cv2.bilateralFilter(input, 9, 75, 75)
    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    return mse, struct_sim

def metric_median(input, truth):

    filtered = cv2.medianBlur(input, 3)
    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    return mse, struct_sim

def metric_wiener(input, truth):

    psf = np.ones((5,5)) / 5**2
    filtered = wiener(input)
    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    return mse, struct_sim

def metric_wavelet(input, truth):

    filtered = denoise_wavelet(input)
    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    return mse, struct_sim

def metric_denoise_tv_chambolle(input, truth):

    filtered = denoise_tv_chambolle(input)
    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    return mse, struct_sim

def metric_denoise_nl_means(input, truth):

    filtered = denoise_nl_means(input)
    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    return mse, struct_sim

def metric_denoise_tv_bregman(input, truth):

    filtered = denoise_tv_bregman(input, weight=0.1, eps=0.0002, max_iter=200, isotropic=False)
    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    return mse, struct_sim

denoiser_nn = Denoiser(
    checkpoint_loc="//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13/model/",
    visible_cuda="0")

freqs = np.fft.fftfreq
def metric_denoise_nn(input, truth):

    preproc_img = denoiser_nn.preprocess(input)
    filtered = denoiser_nn.denoise_crop(preproc_img, preprocess=False, postprocess=False)
    filtered = filtered.clip(0., 1.).reshape(512, 512)

    filtered = filtered.astype(np.float32)
    mse = np.sum((filtered-truth)**2) / input.size
    struct_sim = ssim(filtered, truth, data_range=filtered.max()-filtered.min())

    mse_surface = np.absolute(filtered-truth) #Bad name as abs error surface

    fft_input = np.fft.fftshift(np.absolute(np.fft.fft2(input)))
    fft_profile_orig = radial_profile(fft_input, (fft_input.shape[0]//2,fft_input.shape[1]//2))
    fft_profile_orig /= np.sum(fft_profile_orig)
    
    fft_filtered = np.fft.fftshift(np.absolute(np.fft.fft2(filtered)))
    fft_profile = radial_profile(fft_filtered, (fft_filtered.shape[0]//2,fft_filtered.shape[1]//2))
    fft_profile /= np.sum(fft_profile)

    fft_diff = np.absolute(fft_profile-fft_profile_orig)

    return mse, struct_sim, mse_surface, fft_profile_orig, fft_profile, fft_diff


#metrics = [metric_no_metric,
#           metric_gaussian,
#           metric_bilateral,
#           metric_median,
#           metric_wiener,
#           metric_wavelet,
#           metric_denoise_tv_chambolle,
#           metric_denoise_nl_means,
#           metric_denoise_tv_bregman, 
#           metric_denoise_nn]

metrics = [metric_denoise_nn]
nn_idx = len(metrics)-1

def get_scale():
    return 25+75*np.random.rand()

#img=imread(in_dir+"train/train367.tif", mode='F')
#lq = gen_lq(img, 25)
#for _ in range(10000):
#    disp((np.mean(lq)/np.mean(img))*img)
#    disp(lq)

for i, subset in enumerate([subsets[2]]): #Running over test set
    in_loc = in_dir+subset+"/"
    files = os.listdir(in_loc)

    losses = np.zeros((len(files), len(metrics), 2))
    mse_error_avg = np.zeros((512,512))
    fft_radial_avg_orig = np.zeros((363,))
    fft_radial_avg_filtered = np.zeros((363,))
    fft_diff = np.zeros((363,))
    #losses = np.load(save_dir+subset+"-losses-ssim2.npy")
    for j, file in enumerate(files[:20000]):
        print("{} file {}".format(subset, j))

        try:
            img = imread(in_loc+file, mode='F')
        except:
            img = 0.5*np.ones((512,512))

        lq = gen_lq(img, scale=get_scale())
        img *= np.mean(lq)/np.mean(img)

        for k, metric in enumerate(metrics):
            if not k == nn_idx:
                losses[j, k][0], losses[j, k][1] = metric(lq, img)
            else:
                losses[j, k][0], losses[j, k][1], mse_error_avg_inst, fft_profile_orig, fft_profile_filtered, fft_diff_inst = metric(lq, img)
                mse_error_avg += mse_error_avg_inst
                fft_radial_avg_orig += fft_profile_orig
                fft_radial_avg_filtered += fft_profile_filtered
                fft_diff += fft_diff_inst

        print(losses[j,:])

        if not j%1000:
            np.save(save_dir+subset+"-losses-ssim-nn.npy", losses)
            np.save(save_dir+subset+"-losses-ssim-nn_mse_avg-actually-abs.npy", mse_error_avg)
            np.save(save_dir+subset+"-losses-ssim-nn_fft_avg_orig.npy", fft_radial_avg_orig)
            np.save(save_dir+subset+"-losses-ssim-nn_fft_avg_filtered.npy", fft_radial_avg_filtered)
            np.save(save_dir+subset+"-losses-ssim-nn_fft_diff.npy", fft_diff)

    np.save(save_dir+subset+"-losses-ssim-nn.npy", losses)
    np.save(save_dir+subset+"-losses-ssim-nn_mse_avg-actually-abs.npy", mse_error_avg/20000)
    np.save(save_dir+subset+"-losses-ssim-nn_fft_avg_orig.npy", fft_radial_avg_orig/20000)
    np.save(save_dir+subset+"-losses-ssim-nn_fft_avg_filtered.npy", fft_radial_avg_filtered/20000)
    np.save(save_dir+subset+"-losses-ssim-nn_fft_diff.npy", fft_diff/20000)