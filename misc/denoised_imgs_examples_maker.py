import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 600
fontsize = 10
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize

import matplotlib.mlab as mlab

import glob

import scipy.stats as stats
from scipy.misc import imread

import cv2

def scale0to1(img, min, max):

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

#Width as measured in inkscape
scale = 1.0
width = scale * 2.2 * 3.487
height = 5*(width / 1.618) / 2.2

x_titles = ["Noisy", "Restored", "Ground Truth"]
y_labels = ["y1", "y2", "y3", "y4", "y5"]

files_loc = r'\\flexo.ads.warwick.ac.uk\Shared41\Microscopy\Jeffrey-Ede\models\denoiser-multi-gpu-8\examples\example-'
file_nums = [20, 21, 23, 24]#, 9]

noisy_files = [files_loc+str(num)+'.tif' for num in file_nums]
restored_mse_files = [files_loc+str(num)+'_restored.tif' for num in file_nums]
restored_ssim_files = [files_loc+str(num)+'.tif' for num in file_nums]
truth_files = [files_loc+str(num)+'_truth.tif' for num in file_nums]

set_mins = []
set_maxs = []

for i in range(len(file_nums)):
    img1 = imread(noisy_files[i], mode='F')
    img2 = imread(restored_mse_files[i], mode='F')
    img3 = imread(restored_ssim_files[i], mode='F')
    img4 = imread(truth_files[i], mode='F')
    set_mins.append(np.min([
        np.min(img1), 
        np.min(img2),
        np.min(img3),
        np.min(img4)]))
    set_maxs.append(np.max([
        np.max(img1), 
        np.max(img2),
        np.max(img3),
        np.max(img4)]))

w = h = 512

subplot_cropsize = 64
subplot_prop_of_size = 0.6
subplot_side = int(subplot_prop_of_size*w)
subplot_prop_outside = 0.2
out_len = int(subplot_prop_outside*subplot_side)
side = w+out_len

num_examples = 5-1
f=plt.figure(figsize=(num_examples, 3))
columns = 3
rows = num_examples
for i in range(num_examples):

    j = 1
    img = np.ones(shape=(side,side))
    img[:w, :w] = scale0to1(imread(noisy_files[i], mode='F'), 
                            min=set_mins[i],
                            max=set_maxs[i])
    img[(side-subplot_side):,(side-subplot_side):] = cv2.resize(img[:subplot_cropsize, :subplot_cropsize], 
                                                                (subplot_side, subplot_side), 
                                                                cv2.INTER_CUBIC)
    img = img.clip(0., 1.)
    k = i*columns+j
    ax = f.add_subplot(rows, columns, k)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.ylabel(y_labels[i])
    ax.set_frame_on(False)
    if not i:
        ax.set_title(x_titles[j-1], fontsize=fontsize)

    j = 2
    img = np.ones(shape=(side,side), dtype=np.float32)
    img[:w, :w] = scale0to1(imread(restored_mse_files[i], mode='F'), 
                            min=set_mins[i],
                            max=set_maxs[i])
    img[(side-subplot_side):,(side-subplot_side):] = cv2.resize(img[:subplot_cropsize, :subplot_cropsize], 
                                                                (subplot_side, subplot_side), 
                                                                cv2.INTER_CUBIC)
    img = img.clip(0., 1.)
    k = i*columns+j
    ax = f.add_subplot(rows, columns, k)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    ax.set_frame_on(False)
    if not i:
        ax.set_title(x_titles[j-1], fontsize=fontsize)

    #j = 3
    #img = np.ones(shape=(side,side), dtype=np.float32)
    #img[:w, :w] = scale0to1(imread(restored_ssim_files[i], mode='F'), 
    #                        min=set_mins[i],
    #                        max=set_maxs[i])
    #img[(side-subplot_side):,(side-subplot_side):] = cv2.resize(img[:subplot_cropsize, :subplot_cropsize], 
    #                                                            (subplot_side, subplot_side), 
    #                                                            cv2.INTER_CUBIC)
    #img = img.clip(0., 1.)
    #k = i*columns+j
    #ax = f.add_subplot(rows, columns, k)
    #plt.imshow(img, cmap='gray')
    #plt.xticks([])
    #plt.yticks([])
    #ax.set_frame_on(False)
    #if not i:
    #    ax.set_title(x_titles[j-1], fontsize=fontsize)

    j = 3
    img = np.ones(shape=(side,side), dtype=np.float32)
    img[:w, :w] = scale0to1(imread(truth_files[i], mode='F'), 
                            min=set_mins[i],
                            max=set_maxs[i])
    img[(side-subplot_side):,(side-subplot_side):] = cv2.resize(img[:subplot_cropsize, :subplot_cropsize], 
                                                                (subplot_side, subplot_side), 
                                                                cv2.INTER_CUBIC)
    img = img.clip(0., 1.)
    k = i*columns+j
    ax = f.add_subplot(rows, columns, k)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    ax.set_frame_on(False)
    if not i:
        ax.set_title(x_titles[j-1], fontsize=fontsize)

f.subplots_adjust(wspace=0.07, hspace=-.095)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)
#f.tight_layout()

f.set_size_inches(width, height)

#plt.show()

f.savefig('example_denoising_plot.png', bbox_inches='tight')

#codes = [(7, 2, x+1) for x in range(14)]
#labels = ["Unfiltered", "Gaussian", "Bilateral", "Median", "Wiener", "Wavelet", "Chambolle"]
#data = np.load('//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/train-losses.npy')

#datasets = []
#means = []
#for comp_idx in range(2):
#    for metric_idx in range(7):
#        dataset = data[:num_data_to_use,metric_idx,comp_idx]
        
#        mean = np.mean(dataset[np.isfinite(dataset)])
#        dataset[np.logical_not(np.isfinite(dataset))] = mean

#        if comp_idx == 0:
#            dataset[dataset > mse_x_to] = mse_x_to
#        elif comp_idx == 1:
#            dataset = dataset.clip(0.,1.)

#        means.append(mean)
#        datasets.append(dataset)

#f = plt.figure(1)

#bins_set = []
#density_set = []
#for i in range(len(datasets)):
#    density_set.append(stats.gaussian_kde(datasets[i]))
#    n, bins, patches = plt.hist(np.asarray(datasets[i]).T, num_hist_bins, normed=1, histtype='step')
#    bins_set.append(bins)

#plt.clf()

#integs = []
#maxs = [0., 0.]
#for i in range(7):
#    dens = density_set[i](bins_set[i])
    
#    dens = [den for _, den in sorted(zip(bins_set[i], dens))]
#    bins = sorted(bins_set[i])
#    integ = np.trapz(dens, bins)

#    max = np.max(dens/integ)
#    if max > maxs[0]:
#        maxs[0] = max

#    integs.append(integ)

#for i in range(7, 14):
#    dens = density_set[i](bins_set[i])
    
#    dens = [den for _, den in sorted(zip(bins_set[i], dens))]
#    bins = sorted(bins_set[i])
#    integ = np.trapz(dens, bins)

#    max = np.max(dens/integ)
#    if max > maxs[1]:
#        maxs[1] = max

#    integs.append(integ)

#ax = f.add_subplot(1,2,1)
#for i in range(7):
#    dens = density_set[i](bins_set[i])
#    dens /= integs[i]
#    print(np.sum(dens))
#    print( 0.012 / maxs[0])
#    dens /= maxs[0]

#    #bins_to_use = bins_set[i] < 0.006
#    #bins_not_to_use = np.logical_not(bins_to_use)
#    #bins = np.append(bins_set[i][bins_to_use], 0.008)
#    #dens = np.append(dens[bins_to_use], np.sum(dens[bins_not_to_use]))

#    plt.plot(bins_set[i], dens, linewidth=1., label=labels[i])
#plt.xlabel('Mean Squared Error')
#plt.ylabel('Relative PDF')
#plt.minorticks_on()
#ax.xaxis.set_ticks_position('both')
#ax.yaxis.set_ticks_position('both')
##ax.grid()
##plt.rc('font', family='serif', serif=['Times'])
##plt.rc('text', usetex=False)
##plt.rc('xtick', labelsize=8)
##plt.rc('ytick', labelsize=8)
##plt.rc('axes', labelsize=8)

#plt.legend(loc='upper right', frameon=False)

#ax = f.add_subplot(1,2,2)
#for i in range(7, 14):
#    dens = density_set[i](bins_set[i])
#    dens /= integs[i]
#    print(np.sum(dens))
#    print(1. / maxs[1])
#    dens /= maxs[1]
#    plt.plot(bins_set[i], dens, linewidth=1.)
#plt.xlabel('Structural Similarity Index')
#plt.minorticks_on()
#ax.xaxis.set_ticks_position('both')
#ax.yaxis.set_ticks_position('both')
##ax.grid()
#plt.tick_params()
##plt.rc('font', family='serif', serif=['Times'])
#plt.rc('text', usetex=False)
#plt.rc('xtick', labelsize=8)
#plt.rc('ytick', labelsize=8)
#plt.rc('axes', labelsize=8)

#plt.show()

#for code, data in zip(codes, datasets):
#    subplot_creator(code, data)
