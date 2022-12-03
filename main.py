import matplotlib.pyplot as plt
import numpy as np
import nnfs
import json
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)


# Files with tipe list of tuples, ex: [("file1", 0), ("file2", 1), ...] | Tuple = (filename, expected_output)
def getXy(files_results):
    Xt = []
    yt = []
    for fn, n in files_results:
        f = open(fn, "r")
        text = f.read()
        Xtmp = json.loads(text)
        ytmp = [n] * len(Xtmp)
        print(fn, ":",len(Xtmp), "samples")
        Xt += Xtmp
        yt += ytmp
    return np.array(Xt), np.array(yt)


# ----------------------
files = [("testsCircle", 0), ("tests1", 1), ("tests2", 2)]
X, y = getXy(files)

# ----------------------


# X, y = spiral_data(100, 3)


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.inputs = None
        self.outputs = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def setWeightsBias(self, w, b):
        self.weights = w
        self.biases = b

    def getWeightsBias(self):
        return [self.weights, self.biases]

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Softmax:
    def __init__(self):
        self.dinputs = None
        self.outputs = None

    def forward(self, inputs):
        inputs_sub = inputs - np.max(inputs, axis=1, keepdims=True)
        inputs_exp = np.exp(inputs_sub)
        inputs_exp_normalized = inputs_exp / np.sum(inputs_exp, axis=1, keepdims=True)
        self.outputs = inputs_exp_normalized

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.outputs, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


class ReLU:
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Loss:
    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def acc(self, outputs, y):
        predictions = np.argmax(outputs, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred,
                y_true):  # y_pred = resultados do AI ex: [0.3, 0.6, 0.1], y_true sao os resultados esperados ex: [0, 1, 0]
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Basicamente entre 0 e 1 exclusivo
        if len(y_true.shape) == 1:  # resultados esperados em 1 lista em vez de 1 lista de listas
            correct_confidences = y_pred_clipped[
                range(samples), y_true]  # range(samples)=[0,1,2,3...], y_true = ex: [0,2,1,0...]
        elif len(y_true.shape) == 2:  # caso seja uma matriz...
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        neg_log = -np.log(correct_confidences)
        return neg_log

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.dinputs = None
        self.outputs = None
        self.activation = Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.outputs = self.activation.outputs
        # Calculate and return loss value
        return self.loss.calculate(self.outputs, y_true)

    def acc(self, y):
        predictions = np.argmax(self.outputs, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))
        # Update parameters

    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases ** 2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                          self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) +
                         self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    # Update parameters
    def update_params(self, layer):
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
            # If there is no momentum array for weights
            # The array doesn't exist for biases yet either.
            layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases
        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


steps = 16

np.random.seed(0)


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.activationFunction = ReLU()
        self.endFunction = Softmax()
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.optimizer = Optimizer_Adam(learning_rate=0.015, decay=1e-5)

    def addLayer(self, l):
        self.layers.append(l)
    def getWeigthBias(self):
        w_b = []
        for l in self.layers:
            w_b.append(l.getWeightsBias())
        return w_b
    def learn(self, X, y, times, steps):
        for n in range(times):
            for i in range(0, len(X), steps):
                for lIdx in range(0, len(self.layers)):
                    l = self.layers[lIdx]
                    if lIdx == 0:
                        l.forward(X[i:i + steps])
                    else:
                        l.forward(self.activationFunction.outputs)
                    if lIdx == len(self.layers) - 1:
                        break
                    self.activationFunction.forward(l.outputs)

                # loss_function = Loss_CategoricalCrossentropy()

                # loss = loss_function.calculate(softmax.outputs, y)
                loss = self.loss_activation.forward(self.layers[-1].outputs, y[i:i + steps])

                accuracy = self.loss_activation.acc(y[i:i + steps])

                # if not n % 10:
                #    print(f'epoch: {n}, ' +
                #          f'acc: {accuracy:.3f}, ' +
                #          f'loss: {loss:.3f}, ' +
                #          f'lr: {self.optimizer.current_learning_rate}')
                # Backward pass
                self.loss_activation.backward(self.loss_activation.outputs, y[i:i + steps])

                for lIdx in range(len(self.layers) - 1, -1, -1):
                    l = self.layers[lIdx]
                    if lIdx == len(self.layers) - 1:
                        l.backward(self.loss_activation.dinputs)
                    else:
                        l.backward(self.activationFunction.dinputs)
                    if lIdx == 0:
                        break
                    self.activationFunction.backward(l.dinputs)

                self.optimizer.pre_update_params()
                for l in self.layers:
                    self.optimizer.update_params(l)
                self.optimizer.post_update_params()

    def test(self, X):
        for lIdx in range(0, len(self.layers)):
            l = self.layers[lIdx]
            if lIdx == 0:
                l.forward(X)
            else:
                l.forward(self.activationFunction.outputs)
            if lIdx == len(self.layers) - 1:
                break
            self.activationFunction.forward(l.outputs)
        self.endFunction.forward(self.layers[-1].outputs)
        idx = np.argmax(self.endFunction.outputs, axis=1)
        return idx


nn = NeuralNetwork()
nn.addLayer(Layer(28 ** 2, 64))
nn.addLayer(Layer(64, 3))
nn.learn(X, y, 201, 32) #int(len(X)/25))
# optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)

# ---------------------------
filest = [("tests0After", 0), ("tests1After", 1), ("tests2After", 2)]
X1, y1 = getXy(filest)
# --------------------------

pred = nn.test(X1)
print("pred:", pred)
print("expt:", y1)

accuracy = np.mean(pred == y1)
print(f'accuracy: {accuracy*100:.3f} %')

np.set_printoptions(threshold=np.inf)
# f = open("weightsBias", "w")

# lista = np.array([layer1.getWeightsBias(),layer2.getWeightsBias()])
# np.save("weightsBias", lista)
