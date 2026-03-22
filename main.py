import jax.numpy as jnp
import optax
from data import load_data
from jax import random
from model import Net
from train import train


def main():
    train_images, train_labels, test_images, test_labels = load_data()

    # Initialize model and optimizer
    key = random.PRNGKey(0)

    num_classes = 10
    model = Net(num_classes=num_classes)

    params = model.init(key, train_images[0:1])
    tx = optax.adam(0.001)
    opt_state = tx.init(params)

    num_epochs = 10
    batch_size = 1024

    params, opt_state = train(
        model,
        params,
        tx,
        opt_state,
        train_images,
        train_labels,
        num_classes,
        num_epochs,
        batch_size,
    )

    # Evaluation
    test_logits = model.apply(params, test_images)
    test_predictions = jnp.argmax(test_logits, axis=-1)
    accuracy = jnp.mean(test_predictions == test_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
