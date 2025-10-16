import tensorflow as tf

class Chars74KLoader:
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32, validation_split=0.1):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split

    def load_data(self):
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_split,
            subset='training',
            seed=123
        )
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_split,
            subset='validation',
            seed=123
        )
        class_names = train_dataset.class_names
        test_dataset = val_dataset.unbatch().take(10)
        test_dataset = list(test_dataset)
        return train_dataset, val_dataset, class_names, test_dataset