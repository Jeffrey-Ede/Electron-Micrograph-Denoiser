import numpy as np

data_loc = r'\\flexo.ads.warwick.ac.uk\Shared41\Microscopy\Jeffrey-Ede\denoiser-13-general-stats\test-losses-ssim-nn.npy'
data = np.load(data_loc)

# width as measured in inkscape
scale = 1.0
width = scale * 2.2 * 3.487
height = (width / 1.618) / 2.2
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

datasets = []
means = []
for comp_idx in range(2):
    for metric_idx in range(10):
        dataset = data[:num_data_to_use,metric_idx,comp_idx]
        
        #mean = np.mean(dataset[np.isfinite(dataset)])
        #std_dev = np.std(dataset[np.isfinite(dataset)])
        #max = np.max(dataset[np.isfinite(dataset)])
        #dataset[np.logical_not(np.isfinite(dataset))] = mean

        print(np.sum(dataset > 0.))

        if comp_idx == 0:
            dataset[dataset > mse_x_to] = mse_x_to
        elif comp_idx == 1:
            dataset = dataset.clip(0.,1.)

        mean = np.mean(dataset[np.isfinite(dataset)])
        #if comp_idx == 0 and metric_idx == 0:
        #    for i, n in enumerate(dataset[np.isfinite(dataset)]):
        #        print(i, n)
        std_dev = np.std(dataset[np.isfinite(dataset)])
        max = np.max(dataset[np.isfinite(dataset)])
        dataset[np.logical_not(np.isfinite(dataset))] = mean

        means.append(mean)
        datasets.append(dataset)

        print("Mean: {}, Std Dev: {}, Max: {}".format(mean, std_dev, max))