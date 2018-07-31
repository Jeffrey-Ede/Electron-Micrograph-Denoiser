import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 400
fontsize = 7
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

from PIL import Image

from skimage import exposure

def scale0to1(img):
    
    min = np.min(img)
    max = np.max(img)

    print(min, max)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

#Width as measured in inkscape
scale = 1.0
width = scale * 2.2 * 3.487
height = 1*(width / 1.618) / 2.2

x_titles = ["Low Dose, << 300 counts ppx", "Ordinary Dose, 200-2500 counts ppx"]
y_labels = ["y1", "y2", "y3", "y4", "y5"]

locs = [r'\\flexo.ads.warwick.ac.uk\Shared41\Microscopy\Jeffrey-Ede\denoiser-13-nn-extra-stats\test-losses-ssim-nn_mse_avg-actually-abs.npy',
        r'\\flexo.ads.warwick.ac.uk\Shared41\Microscopy\Jeffrey-Ede\denoiser-13-general-stats\test-losses-ssim-nn_mse_avg-actually-abs.npy']
imgs = [np.load(loc) for loc in locs]

print(np.mean(imgs[0]), np.mean(imgs[1]))

#Image.fromarray(imgs[1]).save('general_abs_err.tif')

set_mins = []
set_maxs = []

for img in imgs:
    set_mins.append(np.min(img))
    set_maxs.append(np.max(img))

w = h = 512

subplot_cropsize = 16
subplot_prop_of_size = 0.6
subplot_side = int(subplot_prop_of_size*w)
subplot_prop_outside = 0.2
out_len = int(subplot_prop_outside*subplot_side)
side = w+out_len

print(imgs[1])

num_examples = 1
f=plt.figure(figsize=(num_examples, 2))
columns = 2
rows = num_examples
d = 256//16#int(307/16)+1
for i in range(num_examples):

    j = 1
    img = np.ones(shape=(side,side))
    tmp = scale0to1(imgs[j-1])
    tmp = tmp[:16,:16]
    sub = np.zeros((256, 256))
    for y in range(256):
        for x in range(256):
            sub[y,x] = tmp[y//d, x//d]

    img[:w, :w] = scale0to1(exposure.equalize_adapthist(imgs[j-1], clip_limit=0.15))
    img[(side-subplot_side):,(side-subplot_side):] = cv2.resize(sub, (307, 307))

    img = img.clip(0., 1.)
    k = i*columns+j
    ax = f.add_subplot(rows, columns, k)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    ax.set_frame_on(False)
    if not i:
        ax.set_title(x_titles[j-1], fontsize=fontsize)

    j = 2
    img = np.ones(shape=(side,side), dtype=np.float32)
    tmp = scale0to1(imgs[j-1])
    tmp = tmp[:16,:16]
    sub = np.zeros((256, 256))
    for y in range(256):
        for x in range(256):
            sub[y,x] = tmp[y//d, x//d]

    img[:w, :w] = scale0to1(exposure.equalize_adapthist(imgs[j-1], clip_limit=0.15))
    img[(side-subplot_side):,(side-subplot_side):] = cv2.resize(sub, (307, 307))
    img = img.clip(0., 1.)
    k = i*columns+j
    ax = f.add_subplot(rows, columns, k)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    ax.set_frame_on(False)
    if not i:
        ax.set_title(x_titles[j-1], fontsize=fontsize)

f.subplots_adjust(wspace=-0.54, hspace=-.095)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)
#f.tight_layout()

f.set_size_inches(width, height)

#plt.show()

f.savefig('abs_err_plot.png', bbox_inches='tight')
