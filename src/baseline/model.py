import numpy as np


class MLP_Hidden2():
    def __init__(self, input_size, hidden_layer1, hidden_layer2, dropout_rate):
        # Layer
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.dropout_rate = dropout_rate

        # Weight
        self.weight1 = np.random.rand(input_size, hidden_layer1)
        self.weight2 = np.random.rand(hidden_layer1, hidden_layer2)
        self.weight3 = np.random.rand(hidden_layer2, 4)

        # Biais
        self.biais1 = np.random.rand(hidden_layer1)
        self.biais2 = np.random.rand(hidden_layer2)
        self.biais3 = np.random.rand(4)

    def forward(self, x: np.ndarray):
        """
        Forward pass
        :param x: numpy array (images)
        :return:
        """
        # 1st layer
        x = self.linear(x=x, weight=self.weight1, biais=self.biais1)
        x = self.relu(x)
        x = self.dropout(x)

        # 2nd layer
        x = self.linear(x=x, weight=self.weight2, biais=self.biais2)
        x = self.relu(x)
        x = self.dropout(x)

        # Output
        output = self.linear(x=x, weight=self.weight3, biais=self.biais3)
        return output

    def backward(self):
        pass

    def linear(self, x, weight, biais) -> np.ndarray:
        return np.dot(x, weight) + biais

    def relu(self, x) -> np.ndarray:
        zero = np.zeros(x.shape, dtype=int)
        return np.maximum(zero, x)

    def dropout(self, x) -> np.ndarray:
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) # Loi de Bernoulli ~ p
        return x * mask

    # TODO:
    def softmax(self, x) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x))

def main():
    np.random.seed(4)
    input_tab = np.array([[1, 0, 1, 2], [1, 3, 0, 1]])
    model = MLP_Hidden2(input_tab.shape[1], 4, 2, 0.5)

    res = model.forward(input_tab)

    print("Input: ", res)

    print("Softmax: ", model.forward(res))


if __name__ == "__main__":
    main()
