import numpy as np
import tensorflow_datasets as tfds


def load_data():

    # Load the MNIST dataset
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()

    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

    # Normalize data
    train_images, train_labels = train_ds["image"], train_ds["label"]
    test_images, test_labels = test_ds["image"], test_ds["label"]

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    return train_images, train_labels, test_images, test_labels
