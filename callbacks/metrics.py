import tensorflow as tf

class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}, "
              f"val_loss={logs.get('val_loss'):.4f}, val_accuracy={logs.get('val_accuracy'):.4f}")
