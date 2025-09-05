import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.6)
    sns.heatmap(cm, annot=True, linewidths=.5)
    plt.show()

def plot_incorrect_fractions(cm):
    incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
    plt.bar(np.arange(len(cm)), incorr_fraction)
    plt.xlabel('True Label')
    plt.ylabel('Fraction of incorrect predictions')
    plt.title('Misclassification Rate per Class')
    plt.show()
