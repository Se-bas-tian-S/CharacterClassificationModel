import tensorflow as tf

augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

def preprocess_and_augment(image, label):
    """Applies rescaling and augmentation."""
    # The image is cast to float32 for the operations
    image = tf.cast(image, tf.float32)
    # Apply augmentations
    image = augmentation_layers(image, training=True)
    # Apply rescaling
    image = image / 255.0
    return image, label

class Chars74KLoader:
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32, validation_split=0.1):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split

    def load_data(self):
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_split,
            subset='both',
            seed=123
        )
        class_names = train_dataset.class_names

        # 2. Cache the datasets after they're loaded from disk
        train_dataset = train_dataset.cache()
        val_dataset = val_dataset.cache()
        
        # 3. Unbatch the dataset to get a stream of individual images
        # This allows for perfect shuffling across epoch boundaries
        train_dataset = train_dataset.unbatch()
        
        # 4. Shuffle the individual images and then re-batch
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(self.batch_size)
        
        # 5. Apply the parallel preprocessing function to the new batches
        train_dataset = train_dataset.map(preprocess_and_augment, num_parallel_calls=AUTOTUNE)

        # 6. Rescale validation data (no shuffling or augmentation needed)
        val_dataset = val_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
        
        # 7. Prefetch for performance
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
        
        return train_dataset, val_dataset, class_names
