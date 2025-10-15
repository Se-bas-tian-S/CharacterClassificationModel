from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from callbacks.metrics import PrintMetricsCallback

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, train_dataset, val_dataset, epochs=20):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        callbacks = [
            ModelCheckpoint(filepath='checkpoint_dir/model_epoch_{epoch:02d}.keras', save_freq='epoch'),
            PrintMetricsCallback()
        ]
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        return history
