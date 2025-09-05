import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
from config import SIZE, NUM_CLASSES, N_SAMPLES_PER_CLASS, CSV_PATH, IMAGE_DIR

def load_and_prepare_data():
    df = pd.read_csv(CSV_PATH)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])

    # Balance data
    dfs = [resample(df[df['label'] == i], replace=True, n_samples=N_SAMPLES_PER_CLASS, random_state=42)
           for i in range(NUM_CLASSES)]
    df_balanced = pd.concat(dfs)

    # Image loading
    image_path_map = {os.path.splitext(os.path.basename(x))[0]: x
                      for x in glob(os.path.join(IMAGE_DIR, '*', '*.jpg'))}
    df_balanced['path'] = df['image_id'].map(image_path_map.get)
    df_balanced['image'] = df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))

    X = np.asarray(df_balanced['image'].tolist()) / 255.0
    Y = to_categorical(df_balanced['label'], num_classes=NUM_CLASSES)

    return train_test_split(X, Y, test_size=0.25, random_state=42)
