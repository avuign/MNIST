import flax.linen as nn


class Net(nn.Module):
    num_classes: int

    def setup(self):
        # define the layers (the "parts" of the ansatz)
        self.conv1 = nn.Conv(features=12, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=12, kernel_size=(3, 3))
        self.fc1 = nn.Dense(features=128)
        self.fc2 = nn.Dense(features=self.num_classes)

    def __call__(self, x):
        # define the forward pass (how data flows through the layers)
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        return x
