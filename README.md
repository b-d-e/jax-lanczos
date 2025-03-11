# Large Scale Spectral Density Estimation for Deep Neural Networks

> Fork of [Google's archived spectral density repository](https://github.com/google/spectral-density), which had a bunch of outdated requirements and was incomptable with newer Jax APIs

This repository contains two implementations of the stochastic Lanczos Quadrature algorithm for deep neural networks as used and described in [Ghorbani, Krishnan and Xiao, _An Investigation into Neural Net Optimization via Hessian Eigenvalue Density_ (ICML 2019)](https://arxiv.org/abs/1901.10159).

To run the example notebooks, please first `pip install tensorflow_datasets`.

## Jax Implementation (by [Justin Gilmer](https://github.com/jmgilmer))
The Jax version is fantastic for fast experimentation (especially in conjunction with [trax](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/trax)). The Jupyter [notebook](https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/jax/mnist_hessian_example.ipynb) demonstrates how to run Lanczos in Jax.

The main function is [`lanczos_alg`](https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/jax/lanczos.py#L27), which returns a tridiagonal matrix and Lanczos vectors. The tridiagonal matrix can then be used to generate spectral densities using [`tridiag_to_density`](https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/jax/density.py#L120).

## Differences between implementations
1. The TensorFlow version performs Hessian-vector product accumulation and the actual Lanczos algorithm in float64, whereas the Jax version performs all calculation in float32.
2. The TensorFlow version targets multi-worker distributed setups, whereas the Jax version targets single worker (potentially multi-GPU) setups.

