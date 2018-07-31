import re
import matplotlib.pyplot as plt
import numpy as np

take_ln = True
moving_avg = True
save = True
save_val = True
window_size = 100
dataset_num = 13
mean_from = 000
remove_repeats = True #Caused by starting from the same counter multiple times
log_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
           str(dataset_num)+"/")
#log_loc = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13-general/"
log_file = log_loc+"log.txt"
val_file = log_loc+"val_log.txt"

switch = False
losses = []
losses_iters = []
with open(log_file, "r") as f:
    for line in f:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)

        for i in range(1, len(numbers), 2):
            
            losses.append(float(numbers[i]))
            losses_iters.append(float(numbers[i-1]))

try:
    switch = False
    val_losses = []
    val_iters = []
    with open(val_file, "r") as f:
        for line in f:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            
            for i in range(1, len(numbers), 2):

                val_losses.append(float(numbers[i]))
                val_iters.append(float(numbers[i-1]))
except:
    print("No val log {}".format(val_file))

def remove_repeated_iters(iter_list, losses):
    iter0 = 0
    scratch = []
    for i, iter in enumerate(iter_list):
        if iter < iter0:
            #Go back to smaller iter and delete from there
            j = i-1
            while iter_list[j] > iter:
                j -= 1
            scratch.append( (j,i-1) )

            print(j,i-1)

        iter0 = iter

    last = 0
    iters_short = []
    losses_short = []
    
    if scratch:
        for s in scratch:
            iters_short += iter_list[last:s[0]]
            losses_short += losses[last:s[0]]
            last = s[1]

        iters_short += iter_list[last:]
        losses_short += losses[last:]
    else:
        iters_short = iter_list
        losses_short = losses

    return np.asarray(iters_short), np.asarray(losses_short)

if remove_repeats:
    losses_iters, losses = remove_repeated_iters(losses_iters, losses)
    val_iters, val_losses = remove_repeated_iters(val_iters, val_losses)


def moving_average(a, n=window_size):
    ret = np.cumsum(np.insert(a,0,0), dtype=float)
    return (ret[n:] - ret[:-n]) / float(n)

losses = moving_average(np.array(losses[:])) if moving_avg else np.array(losses[:])
losses_iters = moving_average(np.array(losses_iters[:])) if moving_avg else np.array(losses[:])
val_losses = moving_average(np.array(val_losses[:])) if moving_avg else np.array(val_losses[:])
val_iters = moving_average(np.array(val_iters[:])) if moving_avg else np.array(val_iters[:])

print(np.mean((losses[mean_from:])[np.isfinite(losses[mean_from:])]))

if save:
    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
                str(dataset_num)+"/log.npy")
    np.save(save_loc, losses)

if save_val:
    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
                str(dataset_num)+"/val_log.npy")
    np.save(save_loc, val_losses)

    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
                str(dataset_num)+"/val_iters.npy")
    np.save(save_loc, val_iters)

plt.plot(losses_iters, np.log(losses) if take_ln else losses)
plt.plot(val_iters, np.log(val_losses) if take_ln else val_losses)
plt.show()
