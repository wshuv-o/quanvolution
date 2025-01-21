

#06:10 PM

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import GradientDescentOptimizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import autograd.numpy as anp

# Training parameters
epochs = 200
batch_size = 4

# Load and prepare CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
n_train = 100
n_test = 40
train_images = train_images[:n_train].astype('float32') / 255.0
test_images = test_images[:n_test].astype('float32') / 255.0
train_labels = to_categorical(train_labels[:n_train], 10)
test_labels = to_categorical(test_labels[:n_test], 10)
train_images = train_images.reshape((n_train, -1))
test_images = test_images.reshape((n_test, -1))
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
test_images = scaler.transform(test_images)

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def first_residual_block(inputs, weights):
    qml.RX(inputs[0], wires=0)
    qml.RY(weights[0], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(inputs[1], wires=0)

    qml.CNOT(wires=[1, 2])
    qml.RY(weights[1], wires=3)

    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))  # Returns outputs from qubit 0 and qubit 1

@qml.qnode(dev)
def second_residual_block(inputs, weights):
    qml.RX(inputs[0], wires=2)
    qml.RY(weights[2], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.RX(inputs[1], wires=2)

    qml.CNOT(wires=[3, 0])
    qml.RY(weights[3], wires=1)

    return qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))  # Returns outputs from qubit 2 and qubit 3

def qrf_network(inputs, weights):
    output1 = first_residual_block(inputs, weights)
    output2 = second_residual_block(inputs, weights)
    combined_output = anp.concatenate(output1 + output2)  # Combine all outputs
    return combined_output  # Shape: (4,)

def quantum_classification_network(inputs, weights):
    outputs = []
    for input_image in inputs:  # Loop over each image in the batch
        output = qrf_network(input_image, weights)
        outputs.append(output)
    return anp.array(outputs)  # Shape: (batch_size, 4)

def loss_function(y_pred, y_true):
    # Clip predictions to prevent log(0)
    y_pred = anp.clip(y_pred, 1e-15, 1 - 1e-15)
    reshaped_pred = y_pred.reshape(batch_size, 40)  # Assuming the input shape is (2, 40)

    # Slice into 4 parts and sum corresponding indices
    summed_predictions = anp.sum(reshaped_pred.reshape(batch_size, 4, 10), axis=1)  # Shape: (2, 10)

    # Average the summed results
    averaged_predictions = summed_predictions / 4  # Shape: (2, 10)

    # Compute log of the averaged predictions
    log_predictions = anp.log(averaged_predictions)

    # Calculate loss
    loss = -anp.sum(y_true * log_predictions) / y_true.shape[0]  # Ensure y_true is shaped correctly
    return loss

def cost_function(weights, batch_inputs, batch_labels):
    predictions = quantum_classification_network(batch_inputs, weights)
    exp_predictions = anp.exp(predictions)
    softmax_predictions = exp_predictions / anp.sum(exp_predictions, axis=1, keepdims=True)
    # print("exp_predictions shape", exp_predictions.shape)
    # print("softmax_predictions shape", softmax_predictions.shape)
    # print("softmax_predictions ", softmax_predictions)
    loss = loss_function(softmax_predictions, batch_labels)
    return loss

num_qubits = 4
num_classes = 10

from pennylane.optimize import AdamOptimizer



#weights = anp.random.randn(num_qubits, num_classes)  # No requires_grad here
weights = anp.random.uniform(low=-0.1, high=0.1, size=(num_qubits, num_classes))

opt = AdamOptimizer(stepsize=0.1) #opt = GradientDescentOptimizer(stepsize=1)
weights = pnp.tensor(weights, requires_grad=True)

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, n_train, batch_size):
        batch_inputs = train_images[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]

        weights = opt.step(lambda w: cost_function(w, batch_inputs, batch_labels), weights)

        batch_loss = cost_function(weights, batch_inputs, batch_labels)
        epoch_loss += batch_loss

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (n_train // batch_size)}")