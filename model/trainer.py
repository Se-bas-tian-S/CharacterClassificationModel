from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, train_dataset, val_dataset, epochs=20):
        # Adam optimizer is used, and the categorical_crossentropy loss function. This
        # loss function is used for Classification tasks. Another commonly used loss function, though for regression
        # tasks, is the mean squared error loss function. It penalizes high deviations from the label value by squaring
        # it.
        # As a metric, accuracy is used. It calculates the percentage of predictions matching the labels. Specifically,
        # during training the accuracy is measured on the training data, and after each batch the accuracy against the
        # validation data. Comparing these scores gives insight to the current state of the model, for example a high
        # accuracy on the training data set but simultaneously a low score on the validation set might indicate
        # overfitting. Another metric would be the R2 score, again, this is used for regression tasks.
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=ModelCheckpoint(filepath='checkpoint_dir/model_epoch_{epoch:02d}.keras', save_freq='epoch')
        )
