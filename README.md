# MNIST

Handwritten digit classification on the MNIST dataset, implemented twice: once with ML libraries, once from scratch.

### What is this project about ?

This is a learning project I built to get familiar with the basics of machine learning. The task is to classify 28×28 grayscale images of handwritten digits (0–9) from the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). I implemented the same problem in two ways to understand what happens under the hood.

### Project structure

#### `mnist_jax/` — Using JAX, Flax, and Optax

This one is adapted from a small workshop on [artificial intelligence for high energy physics](https://indico.global/event/14060/) that I attended at EPFL during the summer of 2025.

A convolutional neural network (CNN) using the JAX ecosystem:
- **JAX** — array operations and automatic differentiation
- **Flax** — neural network layers and model definition
- **Optax** — optimizer (Adam) and loss functions

Architecture: 2 convolutional layers → max pooling → 2 dense layers. Reaches ~95% test accuracy.

#### `mnist_from_scratch/` — From scratch with NumPy only

To understand everything in more details I decided to re-produce the result by coding everything from scratch, just using numpy.

A fully connected network (multilayer perceptron) where everything is implemented by hand:
- Forward pass (matrix multiplications, ReLU activation, softmax)
- Cross-entropy loss
- Backpropagation (chain rule applied layer by layer)
- Vanilla stochastic gradient descent

No ML library is used — only NumPy for array operations. Architecture: 784 → 128 → 10. Reaches ~88% test accuracy.

### How to run

```bash
python -m venv venv
source venv/bin/activate
pip install numpy jax jaxlib flax optax tensorflow tensorflow_datasets
```

For the JAX version:
```bash
cd mnist_jax
python main.py
```

For the from-scratch version:
```bash
cd mnist_from_scratch
python main.py
```

### Dependencies

- NumPy
- JAX / JAXlib (for `mnist_jax`)
- Flax (for `mnist_jax`)
- Optax (for `mnist_jax`)
- TensorFlow / TensorFlow Datasets (data loading only)
