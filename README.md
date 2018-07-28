# Low-Dose Electron Micrograph Denoiser

This repository is for a deep atrous convolutional encoder-decoder trained to remove Poisson noise from low-dose electron micrographs. It outperforms existing methods' average mean squared errors by 24.6% and has a batch size 1 (worst case) inference time of 77.0 ms for 1 GTX 1080 Ti GPU and a 3.4 GHz i7-6700 processor. More details will be found in a paper published on it soon.

The repository contains a checkpoint for the fully trained network, a training script `denoiser-multi-gpu.py` and an inference script `denoiser.py`. The training script is written for multi-GPU training in a distributed setting and the inference script loads the neural network once for repeated inference.

## Architecture

<p align="center">
  <img src="noise-removal-nn.png">
</p>

This network was developed to test how well a deep atrous convolutional encode-decoder can denoise electron micrographs. The answer is pretty well! It improves upon the mean squared error and structural similarity indices of existing methods by 25% and 15%, respectively. It is inspired by networks Google developed for semantic image segmentation.

## Examples

Here are some example applications of the network to noise applied to 512x512 crops from high-quality micrographs. The images are less blurred than they would be by other filters and there are no local artifacts.

<p align="center">
  <img src="examples1.png">
</p>

## Example Usage

This short script is available as `example_denoiser.py` and gives an example of inference where the neural network is loaded once and used to denoise multiple times:

```python
import numpy as np
from denoiser import Denoiser, disp

#Create a 1500x1500 image from random numbers for demonstration
#Try replacing this with your own image!
img = np.random.rand(1500, 1500)

#Initialize the denoising neural network
noise_remover = Denoiser()

#Denoise a 512x512 crop from the image
crop = img[:512,:512]
denoised_crop = noise_remover.denoise_crop(crop)

#Denoise the entire image
denoised_img = noise_remover.denoise(img)

disp(crop) #Crop before denoising
disp(denoised_crop) #Crop after denoising
disp(img) #Image before denoising
disp(denoised_img) #Image after denoising
```

## Download

To get the training and inference scripts, simply copy the files from or clone this repository:

```
git clone https://github.com/Jeffrey-Ede/Electron-Micrograph-Denoiser.git
cd Electron-Micrograph-Denoiser
```

The last saved checkpoint for the fully trained neural network is available [here](https://drive.google.com/open?id=1ehfRekaNUc1NJzjXeyhF3Tv9kOVWt8wN). To use it, change the location in `checkpoint` to the location you save your copy of the network to.

## Dependencies

This neural network was trained using TensorFlow and requires it and other common python libraries. Most of these libraries come with modern python distributions by default. If you don't have some of these libraries, they can be installed using pip or another package manager. 

Libraries you need for both training and inference:

* tensorFlow
* numpy
* cv2
* functools
* itertools
* collections
* six
* os

For training you also need:

* argparse
* random
* scipy
* Image

The network was scripted for python 3.6 using Windows 10. Small adjustments may need to be made for other environments or operating systems.

## Training

To continue training the neural network; end-to-end or to fine-tune it, you will need to adjust some of the variables at the top of the `denoiser-multi-gpu.py` training file. Specifically, variables indicating the location of your datasets and locations to save logs and checkpoints to.


## Training Data

At the momoment, my dataset of 17267 2048x2048 micrograph with mean electron counts of at least 2500 ppx is only available through a request to the Warwick microscopy research technoledgy platform. This can be done by either contacting

* Me at j.m.ede@warwick.ac.uk 
* My PhD supervisor at r.beanland@warwick.ac.uk
* Or [otherwise](https://warwick.ac.uk/fac/sci/physics/research/condensedmatt/microscopy/em-rtp/)

I'm working on making the dataset immediately accessible through Google cloud storage and OpenML and have set this up. However, I'm not able to release it yet due to internal politics. Hopefully this can be resolved soon. In the meantime, making it accessible through contacting us seems to be the best course of action.
