from config import EPOCHS, BATCH_SIZE

def train_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=2
    )
    return history
