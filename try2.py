import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
#from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import LearningRateScheduler
#import matplotlib.pyplot as plt  # Importing matplotlib for plotting
# Load CIFAR-10 dataset
from tensorflow.keras.datasets import cifar10
from tensorflow.image import resize
import tensorflow as tf
from tensorflow.keras import layers, models
# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Subset the dataset
TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SIZE = 20000
SEED = 1000
IMG_SIZE = 50
BATCH_SIZE = 128

# Shuffle and take first TRAIN_SIZE samples
train_indices = np.random.choice(len(x_train), size=TRAIN_SIZE, replace=False)

# Take the first TEST_SIZE samples for test and validation
x_test_small = x_test[:TEST_SIZE]
y_test_small = y_test[:TEST_SIZE]

x_test_small = np.array([resize(image, (IMG_SIZE, IMG_SIZE)) for image in x_test_small])

x_test_small = x_test_small.astype('float32') / 255.0

model1 = tf.keras.models.load_model('resnet50_custom_model.keras', compile=False)

# Compile the model with a new optimizer
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

test_loss, test_accuracy = model1.evaluate(x_test_small, y_test_small, verbose=2)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Get the last layer's output (10 values for each image) using model.predict
last_layer_output = model1.predict(x_test_small, verbose=2)

# Print the last layer output for the first image, for example
print("Last layer output for the first image:", last_layer_output[0])

# Optionally, you can print all outputs for each image
for i, output in enumerate(last_layer_output):
    print(f"Image {i+1} - Last layer output: {output}")

