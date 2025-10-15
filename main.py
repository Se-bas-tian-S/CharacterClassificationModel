from data.loader import Chars74KLoader
from model.architecture import build
from model.trainer import ModelTrainer


if __name__ == "__main__":
    data_dir = "trainingData/englishFnt"
    loader = Chars74KLoader(data_dir)
    train_dataset, val_dataset, class_names = loader.load_data()

    model = build(input_shape=(128, 128, 1), num_classes=len(class_names))
    model.summary()
    trainer = ModelTrainer(model)
    history = trainer.train(train_dataset, val_dataset)

    model.save("final_char_model.keras")
