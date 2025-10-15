from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, 
    Input, RandomRotation, RandomZoom, RandomTranslation, BatchNormalization, Activation)
from tensorflow.keras import mixed_precision


def build(input_shape, num_classes):
    mixed_precision.set_global_policy('mixed_float16')
    model = Sequential([
        Input(input_shape),

        #RandomRotation(0.1),
        #RandomZoom(0.1),
        #RandomTranslation(height_factor=0.1, width_factor=0.1),

        #Rescaling(1.0/255),

        Conv2D(64, (3, 3)),
        #BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3)),
        #BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3)),
        #BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model