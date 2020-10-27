from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.python.keras.callbacks import RemoteMonitor
import tensorboard
from tensorflow.python.keras.utils.np_utils import to_categorical

from classes.Loader import Loader
from classes.Utils import Utils
from tensorflow.keras import layers
from tensorflow import feature_column

df = Loader.load(
    './datasets/private/anonymized.csv',
    usecols=['tld', 'provider', 'fname', 'lname', 'target'],
)

df = df[:15000]
df.dropna(inplace=True)

labels = LabelEncoder().fit_transform(df['target'])
labels = to_categorical(labels)

df['target'] = labels.tolist()

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

batch_size = 32  # A small batch sized is used for demonstration purposes
train_ds = Utils.df_to_dataset(train, batch_size=batch_size)
val_ds = Utils.df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = Utils.df_to_dataset(test, shuffle=False, batch_size=batch_size)


feature_columns = []

for header in ['tld', 'provider', 'fname']:
    feature_col = feature_column.categorical_column_with_vocabulary_list(
        key=header,
        vocabulary_list=df[header].unique()
    )
    nb_categories = df[header].unique().size
    # Best practice is to use the 4th root of the number of possible values
    dimensions = nb_categories ** (1 / float(4))
    feature_embedded = feature_column.indicator_column(feature_col)
    feature_columns.append(feature_embedded)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_ds,
          validation_data=val_ds,
          epochs=20,
          verbose=1,
          )

