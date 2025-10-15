from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from CharacterClassification.callbacks.metrics import PrintMetricsCallback

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=64):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        callbacks = [
            ModelCheckpoint(filepath='model_epoch_{epoch:02d}.h5', save_freq='epoch'),
            PrintMetricsCallback()
        ]
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return history
