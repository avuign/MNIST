import numpy as np
from data import load_data
from model import Net, init_params
from train import compute_loss, train

LAYER_SIZES = [784, 128, 10]
NUM_EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.1


def main():
    print("Starting to download MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_data()
    print("... dataset MNIST downloaded !")

    initial_weights, initial_bias = init_params(LAYER_SIZES)
    model = Net(initial_weights, initial_bias)

    train(model, train_images, train_labels, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)

    # Evaluate
    correct = 0
    for image, label in zip(test_images, test_labels):
        prediction = np.argmax(model(image))
        if prediction == label:
            correct += 1
    accuracy = correct / len(test_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
