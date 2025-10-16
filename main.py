from data.loader import Chars74KLoader
from model.architecture import build
from model.trainer import ModelTrainer
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import numpy as np

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
    train_dataset, val_dataset, class_names, test_dataset = loader.load_data()
    print(class_names)
    test_images = []
    test_labels = []
    for image, label in test_dataset:
        test_images.append(image.numpy().squeeze())
        test_labels.append(label.numpy().argmax())

    model = build(input_shape=(128, 128, 1), num_classes=len(class_names))
    model.summary()
    trainer = ModelTrainer(model)
    history = trainer.train(train_dataset, val_dataset)

    model.save("final_char_model.keras")

    # Loop through each image file
    for i, img in enumerate(test_images):
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.show()
        img_array = np.expand_dims(img, axis=(0, -1))
        img_array = img_array / 255.0  # normalize if your model expects it

        # Run inference
        predictions = model.predict(img_array)

        pred_vector = predictions[0]
        high_conf_indices = np.where(pred_vector > 0.1)[0]
        high_conf_probs = pred_vector[high_conf_indices]

        print(f"Image: {test_labels[i]}")
        print(predictions)
        for index, prob in zip(high_conf_indices, high_conf_probs):
            print(f"Class: {index}: {prob}")

        # Display the image with prediction
        plt.imshow(img, cmap="gray")
        plt.title(f"Prediction: {predictions}")
        plt.axis('off')
        plt.show()