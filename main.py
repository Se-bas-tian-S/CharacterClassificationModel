from data.loader import Chars74KLoader
from model.architecture import build
from model.trainer import ModelTrainer
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    data_dir = "/trainingData/englishFnt"
    loader = Chars74KLoader(data_dir)
    X, y, label_encoder = loader.load_data()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build()
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val)

    model.save("final_char_model.h5")
