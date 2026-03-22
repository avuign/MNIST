import numpy as np


def compute_loss(model, image, label):
    logits, cache = model.forward_with_cache(image)
    exp_logits = np.exp(logits - np.max(logits))
    softmax = exp_logits / np.sum(exp_logits)

    loss = -np.log(softmax[label])

    return loss, softmax, cache


def make_one_hot(label, size):
    one_hot = np.zeros(size)
    one_hot[label] = 1
    return one_hot


def compute_grad(softmax, cache, label, model):
    one_hot = make_one_hot(label, len(softmax))
    grad_weights = []
    grad_bias = []

    n_layers = len(model.weights)
    delta = softmax - one_hot

    for i in range(n_layers - 1, -1, -1):
        a = cache[2 * i]
        grad_weights.append(np.outer(a, delta))
        grad_bias.append(delta)
        if i > 0:
            delta = np.multiply(
                np.matmul(delta, np.transpose(model.weights[i])), (cache[2 * i - 1] > 0)
            )

    return list(reversed(grad_weights)), list(reversed(grad_bias))


def train(model, train_images, train_labels, num_epochs, batch_size, lr):
    for epoch in range(num_epochs):
        for i in range(0, len(train_images[:10000]), batch_size):
            batch = {
                "image": train_images[i : i + batch_size],
                "label": train_labels[i : i + batch_size],
            }

            grads = [np.zeros_like(w) for w in model.weights]
            bias = [np.zeros_like(b) for b in model.bias]

            for image, label in zip(batch["image"], batch["label"]):
                loss, softmax, cache = compute_loss(model, image, label)
                grad_batch, bias_batch = compute_grad(softmax, cache, label, model)

                for j in range(len(model.weights)):
                    grads[j] += grad_batch[j] / batch_size
                    bias[j] += bias_batch[j] / batch_size

            # Update parameters via gradient descent
            for k in range(0, len(model.weights)):
                model.weights[k] -= lr * grads[k]
                model.bias[k] -= lr * bias[k]

        print(f"Epoch {epoch + 1}, Loss: {loss}")
