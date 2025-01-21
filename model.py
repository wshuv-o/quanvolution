
#xx:10 PM

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import autograd.numpy as anp

# Training parameters
epochs = 5
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
    print("first_residual_block")
    print("inputs", inputs)
    print("inp shape:", inputs.shape)
    print("weights", weights)
    print("weights shape:", weights.shape)
    print("abaillaaa", inputs[0])
    qml.RX(inputs[0], wires=0)
    qml.RY(weights[0], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(inputs[1], wires=0)

    qml.CNOT(wires=[1, 2])
    qml.RY(weights[1], wires=3)

    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))  # Returns outputs from qubit 0 and qubit 1

@qml.qnode(dev)
def second_residual_block(inputs, weights):
    print("Second_residual_block")
    print("inputs", inputs)
    print("inp shape:", inputs.shape)
    print("weights", weights)
    print("weights shape:", weights.shape)
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
    reshaped_pred = y_pred.reshape(batch_size, 40)  # Assuming the input shape is (batch_size, 40)

    # Slice into 4 parts and sum corresponding indices
    summed_predictions = anp.sum(reshaped_pred.reshape(batch_size, 4, 10), axis=1)  # Shape: (batch_size, 10)

    # Average the summed results
    averaged_predictions = summed_predictions / 4  # Shape: (batch_size, 10)

    # Compute log of the averaged predictions
    log_predictions = anp.log(averaged_predictions)

    # Calculate loss
    loss = -anp.sum(y_true * log_predictions) / y_true.shape[0]  # Ensure y_true is shaped correctly
    return loss

def cost_function(weights, batch_inputs, batch_labels):
    predictions = quantum_classification_network(batch_inputs, weights)
    exp_predictions = anp.exp(predictions)
    softmax_predictions = exp_predictions / anp.sum(exp_predictions, axis=1, keepdims=True)

    # Debugging outputs
    print("Predictions:", predictions)
    print("Softmax predictions shape:", softmax_predictions.shape)
    print("Softmax predictions:", softmax_predictions)

    loss = loss_function(softmax_predictions, batch_labels)

    # Debugging loss
    print(f"Loss1: {loss}")

    return loss

num_qubits = 4
num_classes = 10

# Initialize weights with a smaller range
weights = anp.random.uniform(low=-0.1, high=0.1, size=(num_qubits, num_classes))

# Use the Adam optimizer for smoother convergence
opt = AdamOptimizer(stepsize=0.01)

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

# Commented out IPython magic to ensure Python compatibility.
import pennylane as qml
# Just like standard NumPy, but with the added benefit of automatic differentiation
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline

"""In this notebook we implement the *Quanvolutional Neural Network*, a quantum machine learning model originally introduced in [Henderson et al. (2019)](https://arxiv.org/abs/1904.04767).

<img src="circuit.png" alt="circuit" width="600"/>

## Introduction

A *Convolutional Neural Network* (**CNN**) is a standard model in (classical) machine learning, especially suitable for image processing. This model is based on the idea of a *convolution layer* where, instead of processing the full input data with a global function, a *local* convolution is applied.
Small local regions are sequentially processed with the same kernel. The results obtained for each region are then associated to different channels of a single output pixel. The union of all the output pixels produces a new image-like object, which can be further processed by additional layers.

One can then consider **quantum variational circuits**, which are quantum algorithms depending on free parameters. These algorithms are trained by a **classical optimization** algorithm that makes queries to the **quantum device**, the optimization being an iterative scheme that searches out better candidates for the parameters with every step. Variational circuits have become popular as a way to think about quantum algorithms for **near-term quantum devices**.

In this notebook we will implement a simplified approach, which will, however, allow us to grasp the idea behind the so-called Quanvolutional Neural Networks (**QNNs**). The scheme is represented in the figure at the top.

1.  A small region of the input image, in our example a $2 \times 2$ square, is embedded into a quantum circuit – this is achieved with parametrized rotations applied to the qubits initialized in the ground state.
2.  A quantum computation, associated to a unitary $U$, is performed on the system – the unitary could be generated by a variational quantum circuit or, more simply, by a *random circuit* as proposed in [Henderson et al. (2019)](https://arxiv.org/abs/1904.04767).
3.  The quantum system is measured, obtaining a list of classical expectation values.
4.  Analogously to a classical convolution layer, each expectation value is mapped to a different channel of a single output pixel.
5.  Iterating the same procedure over different regions, one can scan the full input image, producing an output object which will be structured as a multi-channel image.

**Note** that:
- a fixed non-trainable quantum circuit is used as a "quanvolution" kernel, while the subsequent classical layers are trained for the classification problem of interest;
- the quanvolution can be followed by further quantum layers or by classical layers;
- the **main difference** with respect to a classical convolution is that a quantum circuit can generate highly-complex kernels whose computation could be – at least in principle – classically intractable.

## Setting of hyper-parameters
"""

n_epochs = 30
n_layers = 1
n_train = 100
n_test = 40

SAVE_PATH = "quanvolution/"
PREPROCESS = True           # False --> skip quantum processing and load data from SAVE_PATH
np.random.seed(0)
tf.random.set_seed(0)

"""## MNIST dataset

Here we will use only a small number of training and test images to speedup the evaluation; obviously, better results are achievable using the full dataset.
"""

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Reduce dataset size
train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

# Normalize pixel values within 0 and 1
train_images = train_images / 255
test_images = test_images / 255

# Add extra dimension (for convolution channels)
train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

"""## Quantum circuit as a convolution kernel

Now we initialize a [PennyLane](https://pennylane.ai/) `default.qubit` device – a pure-state qubit simulator – simulating a system of $4$ qubits. The associated `qnode` – an abstract encapsulation of a quantum function, described by a quantum circuit – consists of:

1.  an embedding layer of local $R_y$ rotations (with angles scaled by a factor of $\pi$);
2.  a random circuit of `n_layers`;
3.  a final measurement in the computational basis, estimating $4$ expectation values.
"""

dev = qml.device("default.qubit", wires=4)

# Random circuit parameters
rand_params = np.random.uniform(high=2*np.pi, size=(n_layers, 4))

# To convert the function into a QNode running on dev, we apply the qnode() decorator
@qml.qnode(dev)
def circuit(phi):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi*phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)))

    # Measurement (expect. val.) producing 4 classical outputs
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

"""We now build a function defining the convolution scheme:

1.  the image is divided into squares of $2 \times 2$ pixels;
2.  each square is processed by the quantum circuit;
3.  the $4$ expectation values are mapped into $4$ different channels of a single output pixel.

The process *halves* the resolution of the input image. In the standard CNN-language, this would correspond to a convolution with a $2 \times 2$ *kernel* and a *stride* equal to $2$.
"""

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit"""
    out = np.zeros((14, 14, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Process a 2x2 region with the quantum circuit
            q_results = circuit(
                [image[j, k, 0], image[j, k + 1, 0],
                 image[j + 1, k, 0], image[j + 1, k + 1, 0]]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out

"""## Quantum pre-processing

Since we are not going to train the quantum convolution layer, we apply it as a **pre-processing** layer to the images; then, an entirely "classical" model will be trained and tested on the pre-processed dataset. This procedure will let us avoid unnecessary repetitions of quantum computations.

The pre-processed images are saved in the folder `SAVE_PATH`, thus they can be directly loaded by setting `PREPROCESS = False` – otherwise the quantum convolution is evaluated at each run of the code.
"""

if PREPROCESS == True:
    q_train_images = []
    print("Quantum pre-processing of train images:")
    for idx, img in enumerate(train_images):
        print(f"{idx + 1}/{n_train}        ", end="\r")
        q_train_images.append(quanv(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print("\nQuantum pre-processing of test images:")
    for idx, img in enumerate(test_images):
        print(f"{idx + 1}/{n_test}        ", end="\r")
        q_test_images.append(quanv(img))
    q_test_images = np.asarray(q_test_images)

    # Save pre-processed images
    np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
    np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


# Load pre-processed images
q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
q_test_images = np.load(SAVE_PATH + "q_test_images.npy")

"""Let us visualize the effect of the quantum convolution layer on a batch
of samples:

"""

n_samples = 8
n_channels = 4

fig, axes = plt.subplots(1+n_channels, n_samples, figsize=(10, 10))
for k in range(n_samples):
    axes[0, 0].set_ylabel("Input")
    if k != 0:
        axes[0, k].yaxis.set_visible(False)
    axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")

    # Plot all output channels
    for c in range(n_channels):
        axes[c + 1, 0].set_ylabel(f"Output [ch. {c}]")
        if k != 0:
            axes[c, k].yaxis.set_visible(False)
        axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")

plt.tight_layout()
plt.savefig(f'out_q_layer_{n_samples}_samples.png', dpi=300, facecolor='w')
plt.show()

"""Below each input image, the $4$ output channels generated by the quantum convolution are visualised.

One can clearly notice the **downsampling** of the resolution, along with some local distortion introduced by the quantum kernel; on the other hand, the global shape of the image is preserved, as expected for a convolution layer.

## Hybrid quantum-classical model

After the quanvolution layer, we feed the resulting features into a classical neural network that will be trained to classify the $10$ different digits of the dataset.

We use a *very* simple model: just a **fully connected layer with 10 output nodes** with a final *softmax* activation function.
The model is compiled with a robust and effective *stochastic-gradient-descent* optimizer, **Adam**, and a *cross-entropy* loss function.
"""

def model():
    """Initializes and returns a Keras model to be trained"""
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

"""## Training

We first instantiate the model, then we train and validate it on the dataset that has been *already pre-processed* by the quanvolution.
"""

q_model = model()

q_history = q_model.fit(q_train_images, train_labels,
                        validation_data=(q_test_images, test_labels),
                        batch_size=4, epochs=n_epochs, verbose=2)

"""In order to compare the results achievable *with* and *without* the quanvolution layer, we also initialize a "classical" instance of the model that will be trained and validated on the raw, **not quantum pre-processed** MNIST images."""

c_model = model()

c_history = c_model.fit(train_images, train_labels,
                        validation_data=(test_images, test_labels),
                        batch_size=4, epochs=n_epochs, verbose=2)

"""## Results

We can finally plot the test **accuracy** and **loss** with respect to the number of training epochs.
"""

plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

ax1.plot(q_history.history["val_accuracy"], "-ob", label="With quantum layer")
ax1.plot(c_history.history["val_accuracy"], "-og", label="Without quantum layer")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(q_history.history["val_loss"], "-ob", label="With quantum layer")
ax2.plot(c_history.history["val_loss"], "-og", label="Without quantum layer")
ax2.set_ylabel("Loss")
ax2.set_ylim(top=2.5)
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plt.savefig('accuracy_loss.png', dpi=200, facecolor='w')
plt.show()