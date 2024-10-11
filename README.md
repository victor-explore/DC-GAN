# DC GAN (Deep Convolutional Generative Adversarial Network)

This repository contains an implementation of a DC GAN (Deep Convolutional Generative Adversarial Network) using PyTorch. The DC GAN is trained to generate images based on a given dataset.

## Overview

The DC GAN consists of two main components:

1. Generator: Creates fake images from random noise
2. Discriminator: Distinguishes between real and fake images

The model is trained adversarially, with the generator trying to produce increasingly realistic images to fool the discriminator, while the discriminator improves its ability to detect fake images.

## File Description

- `DC GAN.ipynb`: Jupyter notebook containing the implementation and training process of the DC GAN

## Key Features

- Implements DC GAN architecture using PyTorch
- Uses convolutional layers for both generator and discriminator
- Includes debug information for noise distribution during training
- Trains the model over multiple epochs with batch processing

## Usage

To use this DC GAN:

1. Open the `DC GAN.ipynb` notebook in a Jupyter environment
2. Ensure all required dependencies are installed (PyTorch, etc.)
3. Run the cells in order to train the model and generate images

## Model Details

- The generator takes a 100-dimensional noise vector as input
- The discriminator processes 64x64 pixel images
- Both networks use convolutional layers with batch normalization and LeakyReLU activations

## Training Process

The notebook includes detailed output of the training process, showing:

- Batch progress
- Noise shape for each batch
- Mean and standard deviation of the noise distribution

This information is useful for debugging and ensuring the input noise maintains the expected distribution throughout training.

## Results

After training, the generator should be able to produce realistic images similar to those in the training dataset. The quality of the generated images typically improves over the course of training.

## Note

This implementation is for educational purposes and may require further optimization for production use. Feel free to experiment with different architectures, hyperparameters, or datasets to improve the results.
