import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 400
fontsize = 5
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

import more_itertools

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
num_examples = 5
height = num_examples*(width / 1.618) / 2.2

x_titles = ["Original", "Denoised", "Original", "Denoised"]
y_labels = ["y1", "y2", "y3", "y4", "y5"]

nums = [i for i in range(32)]

loc_orig = r'\\flexo.ads.warwick.ac.uk\Shared39\EOL2100\2100\Users\Jeffrey-Ede\STEM_crops\\'
loc_filtered = r'\\flexo.ads.warwick.ac.uk\Shared39\EOL2100\2100\Users\Jeffrey-Ede\STEM_crops\denoised\img'

nums = [i for i in range(10)]
locs_orig = [loc_orig+str(i)+".tif" for i in nums]
locs_filtered = [loc_filtered+str(i)+".tif" for i in nums]

imgs1 = [imread(loc) for loc in locs_orig]
imgs2 = [imread(loc) for loc in locs_filtered]

#imgs = list(more_itertools.interleave(imgs, imgs_tmp))
imgs = []
for i in range(len(imgs1)):
    imgs.append(imgs1[i])
    imgs.append(imgs2[i])

print(np.mean(imgs[0]), np.mean(imgs[1]))

#Image.fromarray(imgs[1]).save('general_abs_err.tif')

set_mins = []
set_maxs = []

for i in range(len(imgs)//2):

    minimum = min(np.min(imgs[2*i]), np.min(imgs[2*i+1]))
    maximum = max(np.max(imgs[2*i]), np.max(imgs[2*i+1]))

    set_mins.append(minimum)
    set_mins.append(minimum)
    set_maxs.append(maximum)
    set_maxs.append(maximum)

w = h = 512

subplot_cropsize = 64
subplot_prop_of_size = 0.6
subplot_side = int(subplot_prop_of_size*w)
subplot_prop_outside = 0.2
out_len = int(subplot_prop_outside*subplot_side)
side = w+out_len

print(imgs[1])

f=plt.figure(figsize=(num_examples, 4))
columns = 4
rows = num_examples
for i in range(num_examples):

    for j in range(1, 5):
        img = np.ones(shape=(side,side))
        idx = columns*i+j-1
        img[:w, :w] = (imgs[idx]-set_mins[idx]) / (set_maxs[idx]-set_mins[idx])
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
            ax.set_title(x_titles[j-1])#, fontsize=fontsize)

f.subplots_adjust(wspace=-0.69, hspace=0.05)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)
#f.tight_layout()

#f.set_size_inches(width, height)

#plt.show()

f.savefig('stem_examples-denoised2.pdf', bbox_inches='tight')

