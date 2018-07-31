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

import matplotlib.mlab as mlab
import matplotlib.ticker as ticker

import scipy.stats as stats

import re
import pylab

class LogFormatterTeXExponent(pylab.LogFormatter, object):
    """Extends pylab.LogFormatter to use 
    tex notation for tick labels."""
    
    def __init__(self, *args, **kwargs):
        super(LogFormatterTeXExponent, 
              self).__init__(*args, **kwargs)
        
    def __call__(self, *args, **kwargs):
        """Wrap call to parent class with 
        change to tex notation."""
        label = super(LogFormatterTeXExponent, 
                      self).__call__(*args, **kwargs)
        label = re.sub(r'e(\S)0?(\d+)', 
                       r'\\times 10^{\1\2}', 
                       str(label))
        label = "$" + label + "$"
        return label

# width as measured in inkscape
scale = 1.0
ratio = 1.2#1.618
width = scale * 3.487
height = (width / ratio)

take_ln = False
moving_avg = True
window_size = 200

labels = ['Clipping, No SSIM - Train',
          'Clipping, No SSIM - Val',
          'No Clipping, SSIM - Train',
          'No Clipping, SSIM - Val',
          'No Clipping, 2.5 SSIM - Train',
          'No Clipping, 2.5 SSIM - Val',
          'No Clipping, RMSProp, No SSIM - Train',
          'No Clipping, RMSProp, No SSIM - Val']
file_nums = [8, 9, 11, 12]
val_available = [True, True, True, True]
val_file_nums = [8, 9, 11, 12]

#file_nums = [i for i in range(3, 9)]
loc = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"
files = [loc+str(i)+"/log.npy" for i in file_nums]
val_files = [loc+str(i)+"/val_log.npy" for i in val_file_nums]
val_iters_files = [loc+str(i)+"/val_iters.npy" for i in val_file_nums]

datasets = [np.load(file) for file in files]
val_datasets = [np.load(file) 
                for file, available in zip(val_files, val_available) if available]
val_iters_datasets = [np.load(file) 
                      for file, available in zip(val_iters_files, val_available) if available]

print(len(datasets))

trunc_fn = lambda data, max_len: data if data.size <= max_len else data[:max_len]
max_len = datasets[2].size#np.min([data.size for data in datasets])
val_max_len = datasets[-1].size#np.min([data.size for data in val_datasets])
datasets = [datasets[0], 
            datasets[1], 
            datasets[2],
            datasets[3]]

#[trunc_fn(data, max_len) for data in datasets]

def moving_average(a, n=window_size):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

f = plt.figure(1)
ax = f.add_subplot(1,1,1)
val_counter = 0
counter = 0

print(len(datasets))
for i, dataset in enumerate(datasets):
    losses = moving_average(dataset) if moving_avg else dataset
    p = plt.plot(np.linspace(1, losses.size, losses.size), 
                 np.log10(losses), linewidth=1., label=labels[counter])
    counter += 1

    if val_available[i]:
        losses = (moving_average(val_datasets[val_counter]) 
                  if moving_average else val_datasets[val_counter])
        iters = (moving_average(val_iters_datasets[val_counter])
                 if moving_average else val_iters_datasets[val_counter])
        plt.plot(iters, np.log10(losses),
                 linewidth=1., label=labels[counter])
        counter += 1
        val_counter +=1

plt.xlabel('Batches $\\times$10$^3$')
plt.ylabel('Log$_{{10}}$(MSE)')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
#ax.set_x_ticks
ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000.))
ax.xaxis.set_major_formatter(ticks)
#ticks = ticker.FuncFormatter(lambda x, pos: '$10^{{{0:g.4}}}$'.format(x/1000.))
#ax.yaxis.set_major_formatter(ticks)
#ax.set_yscale('log')
#ax.grid()
#plt.rc('font', family='serif', serif=['Times'])
#plt.rc('text', usetex=False)
#plt.rc('xtick', labelsize=8)
#plt.rc('ytick', labelsize=8)
#plt.rc('axes', labelsize=8)

plt.legend(loc='upper right', frameon=False)

f.subplots_adjust(wspace=0., hspace=0.)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

#ax.set_ylabel('Some Metric (in unit)')
#ax.set_xlabel('Something (in unit)')
#ax.set_xlim(0, 3*np.pi)

f.set_size_inches(width, height)

#plt.show()

f.savefig('learning_curves_batch_10.pdf', bbox_inches='tight')


