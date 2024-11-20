import numpy as np


class MLP_Hidden2:
    def __init__(self, input_size, hidden_layer1, hidden_layer2, dropout_rate, batch_size, eta):
        self.input_size = input_size
        self.is_train = True

        # Hyper-parameters
        self.batch_size = batch_size
        self.eta = eta
        self.dropout_rate = dropout_rate

        # Layer
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2

        # Weight
        self.weight1 = np.random.rand(self.input_size, self.hidden_layer1)
        self.weight2 = np.random.rand(self.hidden_layer1, self.hidden_layer2)
        self.weight3 = np.random.rand(self.hidden_layer2, 4)

        # Biais
        self.biais1 = np.random.rand(self.hidden_layer1)
        self.biais2 = np.random.rand(self.hidden_layer2)
        self.biais3 = np.random.rand(4)

        # Output
        self.x = None
        self.hidden1_output = None
        self.hidden2_output = None
        self.y_pred = None

        self.y_true = None

    def evaluation(self):
        self.is_train = False

    def train(self):
        self.is_train = True

    def forward(self, x: np.ndarray):
        """
        Forward pass
        :param x: numpy array (images)
        :return:
        """
        self.x = x
        # 1st layer
        self.hidden1_output = self.linear(x=x, weight=self.weight1, biais=self.biais1)
        self.hidden1_output = self.relu(self.hidden1_output)
        self.hidden1_output = self.dropout(self.hidden1_output)

        # 2nd layer
        self.hidden2_output = self.linear(x=self.hidden1_output, weight=self.weight2, biais=self.biais2)
        self.hidden2_output = self.relu(self.hidden2_output)
        self.hidden2_output = self.dropout(self.hidden2_output)

        # Output layer
        x = self.linear(x=self.hidden2_output, weight=self.weight3, biais=self.biais3)
        self.y_pred = self.softmax(x)
        return self.y_pred

    def loss(self, y_true) -> int:
        self.y_true = y_true
        return - np.sum(y_true * np.log(self.y_pred)) / self.batch_size

    def backward(self):
        # Output layer

        # Calculate output gradient
        gradient_loss = self.y_pred - self.y_true

        # Calculate gradient weight and bias
        d_weight3 = np.dot(self.hidden2_output.T, gradient_loss) / self.batch_size
        d_biais3 = np.sum(gradient_loss, axis=0) / self.batch_size

        # 2nd layer

        # Calculate 2nd hidden gradient
        d_hidden2 = np.dot(gradient_loss, self.weight3.T)
        d_hidden2 = d_hidden2 * (self.hidden2_output > 0)  # Relu derivative

        # Calculate gradient weight and bias
        d_weight2 = np.dot(self.hidden1_output.T, d_hidden2) / self.batch_size
        d_biais2 = np.sum(d_hidden2, axis=0) / self.batch_size

        # 1st layer

        # Calculate 1st hidden gradient
        d_hidden1 = np.dot(d_hidden2, self.weight2.T)
        d_hidden1 = d_hidden1 * (self.hidden1_output > 0)  # Relu derivative

        # Calculate gradient weight and biais
        d_weight1 = np.dot(self.x.T, d_hidden1) / self.batch_size
        d_biais1 = np.sum(d_hidden1, axis=0) / self.batch_size

        # Update all weight and biais
        self.weight1 -= self.eta * d_weight1
        self.biais1 -= self.eta * d_biais1
        self.weight2 -= self.eta * d_weight2
        self.biais2 -= self.eta * d_biais2
        self.weight3 -= self.eta * d_weight3
        self.biais3 -= self.eta * d_biais3

    def linear(self, x, weight, biais) -> np.ndarray:
        return np.dot(x, weight) + biais

    def relu(self, x) -> np.ndarray:
        return np.maximum(0, x)

    def dropout(self, x) -> np.ndarray:
        mask = (np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
                / (1 - self.dropout_rate))  # Loi de Bernoulli ~ p avec rééchelonnement
        return x * mask

    def softmax(self, x) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Exp + prevent overflow
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def main():
    np.random.seed(4)
    input_tab = np.array([[1, 0, 1, 2], [1, 3, 0, 1]])
    true_label = np.array([[0, 0, 1, 0], [1, 0, 0, 0]])
    model = MLP_Hidden2(input_tab.shape[1], 4, 2, 0.5, 1, 0.01)

    y_pred = model.forward(input_tab)

    print("Forward: ", y_pred)

    loss = model.loss(true_label)

    print("Loss: ", loss)

    model.backward()

    y_pred = model.forward(input_tab)

    print("Forward: ", y_pred)

    loss = model.loss(true_label)

    print("Loss: ", loss)

    model.backward()

    y_pred = model.forward(input_tab)

    print("Forward: ", y_pred)

    loss = model.loss(true_label)

    print("Loss: ", loss)


if __name__ == "__main__":
    main()
