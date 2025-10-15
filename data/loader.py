import tensorflow as tf

class Chars74KLoader:
    def __init__(self, data_dir, img_size=(28, 28), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def load_data(self):
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True
        )
        class_names = dataset.class_names
        return dataset, class_names
