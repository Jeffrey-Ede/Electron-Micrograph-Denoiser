import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
fontsize = 10
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['axes.titlepad'] = 10
mpl.rcParams['savefig.dpi'] = 600

import matplotlib.mlab as mlab

import scipy.stats as stats

# width as measured in inkscape
scale = 1.0
ratio = 1.3 # 1.618
width = scale * 2.2 * 3.487
height = 2.2*(width / ratio) / 2.2
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

labels = ["Unfiltered", "Gaussian", "Bilateral", "Median", "Wiener", 
          "Wavelet", "Chambolle TV", "Bregman TV", "NL Means", "Neural Network"]
num = len(labels)
data = np.load('//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/train-losses.npy')
data2 = np.load('//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/test-losses-ssim3.npy')
data_nn = np.load('//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/test-losses-ssim-nn.npy')
data_wiener = np.load('//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/test-losses-ssim-nn-wiener.npy')
codes = [(num, 2, x+1) for x in range(2*num)]

data_general = np.load(r'//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/denoiser-13-general-stats/test-losses-ssim-nn.npy')

datasets = []
datasets_general = []
means = []
means_general = []
for comp_idx in range(2):
    for metric_idx in range(7):

        if metric_idx != 4:
            dataset = data[:num_data_to_use,metric_idx,comp_idx]
        else:
            dataset = data_wiener[:num_data_to_use,0,comp_idx]
        
        mean = np.mean(dataset[np.isfinite(dataset)])
        dataset[np.logical_not(np.isfinite(dataset))] = mean

        if comp_idx == 0:
            dataset[dataset > mse_x_to] = mse_x_to
        elif comp_idx == 1:
            dataset = dataset.clip(0.,1.)

        means.append(mean)
        datasets.append(dataset)

for comp_idx in range(2):
    for metric_idx in range(2):
        dataset = data2[:num_data_to_use,metric_idx,comp_idx]
        
        mean = np.mean(dataset[np.isfinite(dataset)])
        dataset[np.logical_not(np.isfinite(dataset))] = mean

        if comp_idx == 0:
            dataset[dataset > mse_x_to] = mse_x_to
        elif comp_idx == 1:
            dataset = dataset.clip(0.,1.)

        means.append(mean)
        datasets.append(dataset)

for comp_idx in range(2):
    for metric_idx in range(1):
        dataset = data_nn[:num_data_to_use,metric_idx,comp_idx]
        
        mean = np.mean(dataset[np.isfinite(dataset)])
        dataset[np.logical_not(np.isfinite(dataset))] = mean

        if comp_idx == 0:
            dataset[dataset > mse_x_to] = mse_x_to
        elif comp_idx == 1:
            dataset = dataset.clip(0.,1.)

        means.append(mean)
        datasets.append(dataset)

for comp_idx in range(2):
    for metric_idx in range(10):
        dataset = data_general[:num_data_to_use,metric_idx,comp_idx]
        
        mean = np.mean(dataset[np.isfinite(dataset)])
        dataset[np.logical_not(np.isfinite(dataset))] = mean

        if comp_idx == 0:
            dataset[dataset > mse_x_to] = mse_x_to
        elif comp_idx == 1:
            dataset = dataset.clip(0.,1.)

        means_general.append(mean)
        datasets_general.append(dataset)

#Rearrange positions of data
data_tmp = datasets_general[8]
datasets_general[8] = datasets_general[7]
datasets_general[7] = data_tmp

data_tmp = datasets_general[16]
datasets_general[16] = datasets_general[17]
datasets_general[17] = data_tmp
del data_tmp

mean_tmp = means_general[8]
means_general[8] = means_general[7]
means_general[7] = mean_tmp

mean_tmp = means_general[16]
means_general[16] = means_general[17]
means_general[17] = mean_tmp
del mean_tmp

datasets = (datasets[:7] + datasets[14:16] + datasets[18:19] + 
            datasets[7:14] +datasets[16:18] + datasets[19:20])

datasets.extend(datasets_general)
means.extend(means_general)

f, big_axes = plt.subplots( figsize=(15.0, 15.0),nrows=2, ncols=1, sharey=True) 

titles = ["Low Dose, << 300 counts ppx", "Ordinary Dose, 200-2500 counts ppx"]
for row, big_ax in enumerate(big_axes):
    big_ax.set_title(titles[row], fontsize=fontsize)

    # Turn off axis lines and ticks of the big subplot 
    # obs alpha is 0 in RGBA string!
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False
#f.set_facecolor('w')

print(np.min(datasets[12]), np.max(datasets[12]))
print(np.min(datasets[13]), np.max(datasets[13]))
print(np.min(datasets[14]), np.max(datasets[14]))
print(np.min(datasets[15]), np.max(datasets[15]))
print(np.min(datasets[16]), np.max(datasets[16]))
print(np.min(datasets[17]), np.max(datasets[17]))

def subplot_creator(loc, data):
    plt.subplot(loc[0], loc[1], loc[2])

    # the histogram of the data
    n, bins, patches = plt.hist(data, 30, normed=1, facecolor='grey', edgecolor='black', alpha=0.75, linewidth=1)

    # add a 'best fit' line
    #y = mlab.normpdf( bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    #plt.xlabel('Smarts')
    #plt.ylabel('Probability')
    #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    #plt.grid(True)

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=False)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

bins_set = []
density_set = []
for i in range(len(datasets)):
    density_set.append(stats.gaussian_kde(datasets[i]))
    n, bins, patches = plt.hist(np.asarray(datasets[i]).T, num_hist_bins, normed=1, histtype='step')
    bins_set.append(bins)

#plt.clf()

integs = []
maxs = [0., 0., 0., 0.]
for i in range(num):
    dens = density_set[i](bins_set[i])
    
    dens = [den for _, den in sorted(zip(bins_set[i], dens))]
    bins = sorted(bins_set[i])
    integ = np.trapz(dens, bins)

    max = np.max(dens/integ)
    if max > maxs[0]:
        maxs[0] = max

    integs.append(integ)

for i in range(num, 2*num):
    dens = density_set[i](bins_set[i])
    
    dens = [den for _, den in sorted(zip(bins_set[i], dens))]
    bins = sorted(bins_set[i])
    integ = np.trapz(dens, bins)

    max = np.max(dens/integ)
    if max > maxs[1]:
        maxs[1] = max

    integs.append(integ)

for i in range(2*num, 3*num):
    dens = density_set[i](bins_set[i])
    
    dens = [den for _, den in sorted(zip(bins_set[i], dens))]
    bins = sorted(bins_set[i])
    integ = np.trapz(dens, bins)

    max = np.max(dens/integ)
    if max > maxs[2]:
        maxs[2] = max

    integs.append(integ)

for i in range(3*num, 4*num):
    dens = density_set[i](bins_set[i])
    
    dens = [den for _, den in sorted(zip(bins_set[i], dens))]
    bins = sorted(bins_set[i])
    integ = np.trapz(dens, bins)

    max = np.max(dens/integ)
    if max > maxs[3]:
        maxs[3] = max

    integs.append(integ)

print("Maxs: ", maxs)
ax = f.add_subplot(2,2,1)
for i in range(num):
    dens = density_set[i](bins_set[i])
    dens /= integs[i]
    print(np.sum(dens))
    dens /= maxs[0]

    #bins_to_use = bins_set[i] < 0.006
    #bins_not_to_use = np.logical_not(bins_to_use)
    #bins = np.append(bins_set[i][bins_to_use], 0.008)
    #dens = np.append(dens[bins_to_use], np.sum(dens[bins_not_to_use]))

    select = bins_set[i] < 0.0045
    lw = 1 if not i%num == num-1 else 2
    ls = '--' if not i%num else '-'
    plt.plot(bins_set[i][select], dens[select], linewidth=lw, label=labels[i],linestyle=ls)
plt.xlabel('Mean Squared Error')
plt.ylabel('Relative PDF')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
#ax.grid()
#plt.rc('font', family='serif', serif=['Times'])
#plt.rc('text', usetex=False)
#plt.rc('xtick', labelsize=8)
#plt.rc('ytick', labelsize=8)
#plt.rc('axes', labelsize=8)

plt.legend(loc='upper right', frameon=False)

ax = f.add_subplot(2,2,2)
for i in range(num, 2*num):
    dens = density_set[i](bins_set[i])
    dens /= integs[i]
    print(np.sum(dens))
    print(1. / maxs[1])
    dens /= maxs[1]
    lw = 1 if not i%num == num-1 else 2
    ls = '--' if not i%num else '-'
    plt.plot(bins_set[i], dens, linewidth=lw, linestyle=ls)
plt.xlabel('Structural Similarity Index')
plt.ylabel('Relative PDF')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
#ax.grid()
plt.tick_params()
##plt.rc('font', family='serif', serif=['Times'])
#plt.rc('text', usetex=False)
#plt.rc('xtick', labelsize=8)
#plt.rc('ytick', labelsize=8)
#plt.rc('axes', labelsize=8)

ax = f.add_subplot(2,2,3)
for i in range(2*num, 3*num):
    dens = density_set[i](bins_set[i])
    dens /= integs[i]
    print(np.sum(dens))
    print(1. / maxs[2])
    dens /= maxs[2]
    select = bins_set[i] < 0.0012
    lw = 1 if not i%num == num-1 else 2
    ls = '--' if not i%num else '-'
    plt.plot(bins_set[i][select], dens[select], linewidth=lw,linestyle=ls)
plt.xlabel('Mean Squared Error')
plt.ylabel('Relative PDF')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
#ax.grid()
plt.tick_params()

ax = f.add_subplot(2,2,4)
for i in range(3*num, 4*num):
    dens = density_set[i](bins_set[i])
    dens /= integs[i]
    print(np.sum(dens))
    print(1. / maxs[3])
    dens /= maxs[3]
    lw = 1 if not i%num == num-1 else 2
    ls = '--' if not i%num else '-'
    plt.plot(bins_set[i], dens, linewidth=lw,linestyle=ls)
plt.xlabel('Structural Similarity Index')
plt.ylabel('Relative PDF')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
#ax.grid()
plt.tick_params()

#plt.show()

#for code, data in zip(codes, datasets):
#    subplot_creator(code, data)

f.subplots_adjust(wspace=0.18, hspace=0.26)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

#ax.set_ylabel('Some Metric (in unit)')
#ax.set_xlabel('Something (in unit)')
#ax.set_xlim(0, 3*np.pi)

f.set_size_inches(width, height)

#plt.show()

f.savefig('plot.png', bbox_inches='tight', )
