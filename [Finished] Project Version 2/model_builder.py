from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from config import SIZE, NUM_CLASSES

def build_model():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
    return model
