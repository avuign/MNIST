import jax
import jax.numpy as jnp
import optax
from jax import grad


def compute_loss(params, model, images, labels, num_classes):
    logits = model.apply(params, images)
    one_hot = jax.nn.one_hot(labels, num_classes)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))


def train(
    model,
    params,
    tx,
    opt_state,
    train_images,
    train_labels,
    num_classes,
    num_epochs,
    batch_size,
):
    for epoch in range(num_epochs):
        for i in range(0, len(train_images[:10000]), batch_size):
            batch = {
                "image": train_images[i : i + batch_size],
                "label": train_labels[i : i + batch_size],
            }

            # Compute loss
            loss = compute_loss(
                params, model, batch["image"], batch["label"], num_classes
            )

            # Compute gradients
            grads = grad(compute_loss)(
                params, model, batch["image"], batch["label"], num_classes
            )

            # Update parameters
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        print(f"Epoch {epoch + 1}, Loss: {loss}")
    return params, opt_state
