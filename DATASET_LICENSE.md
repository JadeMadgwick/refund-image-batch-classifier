# Dataset attribution and licensing note

This prototype uses **Fashion-MNIST**, a dataset of Zalando article images published by Han Xiao, Kashif Rasul, and Roland Vollgraf. It contains 60,000 28x28 grayscale training images and 10,000 test images across 10 clothing/accessory classes. The dataset is distributed under the MIT license according to the Keras dataset documentation.

- Official repository: https://github.com/zalandoresearch/fashion-mnist
- Paper: Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747. https://arxiv.org/abs/1708.07747
- Dataset API/documentation: https://keras.io/api/datasets/fashion_mnist/

The dataset files are downloaded from the publicly documented TensorFlow/Keras dataset mirror during the first run and are not bundled in this repository. This repository contains code to download and parse them.
