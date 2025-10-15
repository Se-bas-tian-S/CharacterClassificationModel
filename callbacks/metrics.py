import tensorflow as tf

class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}:")
        print(f" Training Accuracy: {logs['accuracy']:.4f}")
        print(f" Validation Accuracy: {logs['val_accuracy']:.4f}")
        print(f" Training Loss: {logs['loss']:.4f}")
        print(f" Validation Loss: {logs['val_loss']:.4f}")
