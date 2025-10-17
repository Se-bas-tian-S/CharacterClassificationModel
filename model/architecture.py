from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling,
    Input, RandomRotation, RandomZoom, RandomTranslation, BatchNormalization, Activation)
from tensorflow.keras import mixed_precision


def build(input_shape, num_classes):
    #mixed_precision.set_global_policy('mixed_float16')
    model = Sequential([
        # Input layer specifies the input dimensions of the first layer
        Input(input_shape),

        # Data augmentation, images get rotated by -36° to 36°, zoomed to 90% to 110% original size and offset by
        # 10% of the height and width
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomTranslation(height_factor=0.1, width_factor=0.1),

        # Rescaling the pixel values from 0-255 to 0-1
        Rescaling(1.0/255),

        # Convolution layer. Scans the image with a sliding window approach in 3x3 pixel windows for each of the filters
        # Each filter has adjustable weights that are being updated during training to individually recognize relevant
        # information.
        Conv2D(16, (3, 3)),
        # Normalization layer. Trained without it first, and actually got better results. Is used to generalize better,
        # sometimes in addition, mostly instead of the Dropout Layer. Goes through the entire batch and normalizes the
        # output of the Convolution layer based on the mean and variation of the outputs of the batch
        BatchNormalization(),
        # Relu activation function. Basically maps all negative values to 0 and keeps all positve values as is. That
        # counters the vanishing gradient problem since the gradient of every positive value is 1. This is important
        # during the backpropagation step when updating the weights.
        Activation('relu'),
        # Really not sure what this layer exactly does. Another window slide, this time over the filter feature maps.
        # Supposed to make the learned features positionally independent. No idea how exactly that works though.
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Flattens the output of the previous layers in shape (height, width, filters) to a 1d vector to be used by the
        # Dense layer afterwards.
        Flatten(),

        # Dense layers with trainable weights, used to convert the information of the output of the convolution layers
        # into an output class. The last dense layer returns a num_classes dimensional vector, the applied softmax
        # function transforms the values so that they add up to 1, resulting in a probability output.
        Dense(128, activation='relu'),
        # Optional Dropout layer used to prevent overfitting by masking a percentage of the inputs for the last Dense
        # layer. Usually either Batch Normalization or a Dropout is used.
        #Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model