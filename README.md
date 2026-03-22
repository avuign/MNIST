# MNIST_CNN_JAX

Handwritten digit classification with a convolutional neural network in JAX/Flax.

### What is this project about ?

This is a learning project I built to get familiar with the basics of machine learning. The goal was to build a neural network model and train it to recognize handwritten digits (the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)). This is not production code — it is an educational exercise.

The model is a small convolutional neural network (CNN) trained with [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax). It reaches ~95% accuracy on the test set after 10 epochs of training on 10,000 images.

### Project structure

- `data.py` — loads and normalizes the MNIST dataset via `tensorflow_datasets`
- `model.py` — defines the CNN architecture (2 conv layers, 2 dense layers)
- `train.py` — loss function (cross-entropy) and training loop
- `main.py` — orchestrates the pipeline: load data → init model → train → evaluate

### How to run

```bash
python -m venv venv
source venv/bin/activate
pip install jax jaxlib flax optax tensorflow tensorflow_datasets
python main.py
```

### Dependencies

- JAX / JAXlib
- Flax
- Optax
- TensorFlow / TensorFlow Datasets
