import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = tf.keras.models.load_model("C:\\Users\\spalls\\Documents\\PythonProjects\\MachineLearningModule\\CharacterClassification\\pc_hand_char_model.keras")
    print(model.input_shape)

    # Define the folder containing images
    image_folder = "exampleImages"

    # Get list of image files in the folder
    image_files = [f for f in os.listdir(image_folder)]

    # Loop through each image file
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)

        # Load and preprocess the image
        img = image.load_img(img_path, color_mode="grayscale", target_size=(128, 128))  # adjust size to match your model's input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=(0, -1))

        # Run inference
        predictions = model.predict(img_array)

        pred_vector = predictions[0]
        high_conf_indices = np.where(pred_vector > 0.1)[0]
        high_conf_probs = pred_vector[high_conf_indices]

        print(f"Image: {img_file}")
        for index, prob in zip(high_conf_indices, high_conf_probs):
            print(f"Class: {index+1}: {prob}")

        # Display the image with prediction
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.show()
