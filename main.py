from data.loader import Chars74KLoader
from model.architecture import build
from model.trainer import ModelTrainer
import tensorflow as tf

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print(f"GPUs detected: {len(gpus)}. Using GPU for training.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Using CPU for training.")

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Is Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    data_dir = "trainingData/Fnt"
    loader = Chars74KLoader(data_dir)
    train_dataset, val_dataset, class_names = loader.load_data()

    model = build(input_shape=(128, 128, 1), num_classes=len(class_names))
    model.summary()
    trainer = ModelTrainer(model)
    history = trainer.train(train_dataset, val_dataset)

    model.save("pc_hand_char_model.keras")