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
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Subset the dataset
TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SIZE = 20000
SEED = 1000
IMG_SIZE = 50
BATCH_SIZE = 128

# Shuffle and take first TRAIN_SIZE samples
train_indices = np.random.choice(len(x_train), size=TRAIN_SIZE, replace=False)
x_train_small = x_train[train_indices]
y_train_small = y_train[train_indices]

# Take the first TEST_SIZE samples for test and validation
x_test_small = x_test[:TEST_SIZE]
y_test_small = y_test[:TEST_SIZE]

x_val_small = x_test[TEST_SIZE:TEST_SIZE + VALIDATION_SIZE]
y_val_small = y_test[TEST_SIZE:TEST_SIZE + VALIDATION_SIZE]

x_train_small = np.array([resize(image, (IMG_SIZE, IMG_SIZE)) for image in x_train_small])
x_val_small = np.array([resize(image, (IMG_SIZE, IMG_SIZE)) for image in x_val_small])
x_test_small = np.array([resize(image, (IMG_SIZE, IMG_SIZE)) for image in x_test_small])

# Normalize the resized images
x_train_small = x_train_small.astype('float32') / 255.0
x_val_small = x_val_small.astype('float32') / 255.0
x_test_small = x_test_small.astype('float32') / 255.0

print(f'x_train_small shape: {x_train_small.shape}')
print(f'y_train_small shape: {y_train_small.shape}')



# Identity Block
def identity_block(X, filters):
    f1, f2, f3 = filters
    X_copy = X

    # 1st Layer
    X = layers.Conv2D(filters=f1, kernel_size=(1,1), strides=(1,1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # 2nd Layer
    X = layers.Conv2D(filters=f2, kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # 3rd Layer
    X = layers.Conv2D(filters=f3, kernel_size=(1,1), strides=(1,1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)

    # Add the Skip Connection
    X = layers.Add()([X, X_copy])
    X = layers.Activation('relu')(X)

    return X

# Convolutional Block
def conv_blocks(X, filters, s=2):
    f1, f2, f3 = filters
    X_copy = X

    # 1st Layer
    X = layers.Conv2D(filters=f1, kernel_size=(1,1), strides=(s,s), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # 2nd Layer
    X = layers.Conv2D(filters=f2, kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # 3rd Layer
    X = layers.Conv2D(filters=f3, kernel_size=(1,1), strides=(1,1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)

    # Adjust skip connection
    X_copy = layers.Conv2D(filters=f3, kernel_size=(1,1), strides=(s,s), padding='valid')(X_copy)
    X_copy = layers.BatchNormalization(axis=3)(X_copy)

    # Add the Skip Connection
    X = layers.Add()([X, X_copy])
    X = layers.Activation('relu')(X)

    return X

# Modified ResNet50 architecture to output 4 values
def ResNet50():
    X_input = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    X = layers.ZeroPadding2D((3,3))(X_input)

    # Stage Conv1
    X = layers.Conv2D(64, (7,7), strides=(2,2))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3,3), strides=(2,2))(X)

    # Stage Conv2_x
    X = conv_blocks(X, filters=[64,64,256], s=1)
    X = identity_block(X, filters=[64,64,256])
    X = identity_block(X, filters=[64,64,256])

    # Stage Conv3_x
    X = conv_blocks(X, filters=[128,128,512], s=2)
    X = identity_block(X, filters=[128,128,512])
    X = identity_block(X, filters=[128,128,512])
    X = identity_block(X, filters=[128,128,512])

    # Stage Conv4_x
    X = conv_blocks(X, filters=[256,256,1024], s=2)
    X = identity_block(X, filters=[256,256,1024])
    X = identity_block(X, filters=[256,256,1024])
    X = identity_block(X, filters=[256,256,1024])
    X = identity_block(X, filters=[256,256,1024])
    X = identity_block(X, filters=[256,256,1024])

    # Stage Conv5_x
    X = conv_blocks(X, filters=[512,512,2048], s=2)
    X = identity_block(X, filters=[512,512,2048])
    X = identity_block(X, filters=[512,512,2048])

    # Dimensionality Reduction to 4 Values
    # Instead of using average pooling, we will go down to 4 values using a dense layer after flattening
    # --------------------------------------------------------
    # X = layers.GlobalAveragePooling2D()(X)  # This flattens the feature map to a vector of 2048 features
    # X = layers.Dense(10, activation='linear')(X)  # Reduce to 4 output values

    X = layers.AveragePooling2D((2,2))(X)
    X = layers.Flatten()(X)
    X = layers.Dense(10, activation='softmax', kernel_initializer='he_normal')(X)

    model = models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

model = ResNet50()
model.summary()

from tensorflow.keras.callbacks import LearningRateScheduler
import keras

# Define learning rate schedule function
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 40:
        lr *= 0.5e-3
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

model = ResNet50()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f'x_train_small shape: {x_train_small.shape}')
print(f'y_train_small shape: {y_train_small.shape}')

h = model.fit(
                x=x_train_small, y=y_train_small,
                epochs=50,
                batch_size=128,
                validation_data=(x_val_small, y_val_small),
                steps_per_epoch=len(x_train_small) // 128,
                validation_steps=len(x_val_small) // 128,
                callbacks=[lr_scheduler],
                verbose=2
             )

# Save the entire model in the new Keras format
model.save('resnet50_custom_model_v2.keras')
model.save('resnet50_custom_model_v2.h5')

#model.save('/content/drive/MyDrive/resnet50_custom_model.keras')

