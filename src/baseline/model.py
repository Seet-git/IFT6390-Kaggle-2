import numpy as np


class MLP_Hidden1:
    def __init__(self, input_size: int, hidden_layer: int, eta: float):
        self.input_size = input_size
        self.is_train = True

        # Hyper-paramètres
        self.eta = eta

        # Couche cachée
        self.hidden_layer = hidden_layer

        # Initialisation des poids avec la formule de Xavier
        self.weight1 = np.random.randn(input_size, hidden_layer) * np.sqrt(2 / self.input_size)
        self.weight2 = np.random.randn(hidden_layer, 4) * np.sqrt(2 / self.input_size)
        # Biais
        self.biais1 = np.zeros((1, hidden_layer))
        self.biais2 = np.zeros((1, 4))

        # Sorties intermédiaires
        self.x = None
        self.hidden_output = None
        self.y_pred = None
        self.y_true = None

    def eval(self):
        self.is_train = False

    def train(self):
        self.is_train = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        :param x:
        :return:
        """
        self.x = x

        # 1st layer
        self.hidden_output = self.linear(x, self.weight1, self.biais1)

        self.hidden_output = self.relu(self.hidden_output)

        # Output layer
        x = self.linear(self.hidden_output, self.weight2, self.biais2)

        return self.softmax(x)

    def loss(self, y_true, y_pred) -> float:
        self.y_true = y_true
        self.y_pred = y_pred
        epsilon = 1e-10  # Éviter log(0)
        return -np.sum(y_true * np.log(y_pred + epsilon)) / y_true.shape[0]

    def backward(self):
        # Gradient de la couche de sortie
        gradient_loss = self.y_pred - self.y_true
        d_weight2 = np.dot(self.hidden_output.T, gradient_loss)
        d_biais2 = np.sum(gradient_loss, axis=0)

        # Gradient de la couche cachée
        d_hidden = np.dot(gradient_loss, self.weight2.T)
        d_hidden = d_hidden * (self.hidden_output > 0)  # Dérivée de ReLU
        d_weight1 = np.dot(self.x.T, d_hidden)
        d_biais1 = np.sum(d_hidden, axis=0)

        # Mise à jour des poids et biais
        self.weight1 -= self.eta * d_weight1
        self.biais1 -= self.eta * d_biais1
        self.weight2 -= self.eta * d_weight2
        self.biais2 -= self.eta * d_biais2

    def linear(self, x, weight, biais) -> np.ndarray:
        return np.dot(x, weight) + biais

    def relu(self, x) -> np.ndarray:
        return np.maximum(0, x)

    def softmax(self, x) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prévenir les dépassements
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
