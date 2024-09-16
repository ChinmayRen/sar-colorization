# sar-colorization
This project focuses on colorizing Synthetic Aperture Radar (SAR) images to enhance their interpretability. The model uses a pix2pix architecture to convert grayscale SAR images into colored images by learning from pairs of SAR and optical images.

The model uses a pix2pix architecture, which is a type of Conditional Generative Adversarial Network (cGAN). The pix2pix model is designed for image-to-image translation tasks, making it suitable for transforming grayscale SAR images into their colored counterparts.

## Pre - Processing:
We found that the given black & white images were very noisy (unlike high resolution images usually obtained from SAR imaging)

Therefore,
We had turn our Colored Images into LAB color space where the 3 channels are:
L - Luminosity (how black or white an image is)
a - Red Green channel(how reddish or greenish an image is)
b - Blue Yellow channel(how 

And we got our training data by extracting the L channel (which is basically black & white image)

## Model Structure

This project implements a Generative Adversarial Network (GAN) for image-to-image translation tasks. The model consists of two main components: a generator and a discriminator.

### Generator

The generator follows a U-Net architecture:

- **Input**: 256x256x1 (grayscale image)
- **Encoder (Downsampling)**:
  - 8 downsampling layers
  - Filters: 64, 128, 256, 512, 512, 512, 512, 512
  - Each layer: Conv2D -> BatchNorm -> LeakyReLU
- **Decoder (Upsampling)**:
  - 7 upsampling layers
  - Filters: 512, 512, 512, 512, 256, 128, 64
  - Each layer: Conv2DTranspose -> BatchNorm -> ReLU
  - Dropout (0.5) applied to first 3 upsampling layers
- **Output**: 256x256x3 (RGB image)
- Skip connections between encoder and decoder layers

- What the U-Net architecture does is:
  - It's design allows for "skips" or connections between the downsampling & upsampling layers,
  which allows the model to retain many low-level features such as colors or textures(extremely important for the purpose of our model)
  - It also allows for retention of high level features since the image shall not lose those features while undergoing the process of colorization.
  - And ultimately, it allows the generator to learn the mapping from grayscale (or L channel) to the colorized image (RGB)

### Discriminator

The discriminator follows a PatchGAN architecture:

- **Input**: Concatenated 256x256x1 (input image) and 256x256x3 (target/generated image)
- **Structure**:
  - 3 downsampling layers (64, 128, 256 filters)
  - 2 convolutional layers (512, 1 filters)
- **Output**: 30x30x1 patch, determining if each 70x70 patch of the input is real or fake

- What PatchGAN essentially does is:

Instead of looking at the whole image at once, PatchGAN looks at small patches of the image.
It decides if each patch looks real or fake.
This helps the model focus on getting the small details right, not just the overall image.

### Loss Functions

- **Discriminator Loss**: Binary cross-entropy
- **Generator Loss**: Combination of adversarial loss (binary cross-entropy) and L1 loss

### Optimizers

Both generator and discriminator use Adam optimizer with learning rate 2e-4 and beta1 0.5.

This architecture enables the model to learn a mapping from input images to output images, guided by both pixel-wise accuracy (L1 loss) and realism (adversarial loss).


DATASET LINK: https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain
