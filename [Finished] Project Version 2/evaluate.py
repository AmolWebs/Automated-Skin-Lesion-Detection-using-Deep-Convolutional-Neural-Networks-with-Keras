import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils import plot_confusion_matrix, plot_incorrect_fractions

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm)
    plot_incorrect_fractions(cm)

def plot_results(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.plot(epochs, history.history['loss'], 'y', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, history.history['acc'], 'y', label='Training acc')
    plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
