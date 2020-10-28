import datetime

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

# Prepare TLD column as a feature
tld_col = feature_column.categorical_column_with_vocabulary_list(key='tld', vocabulary_list=df['tld'].unique())
nb_tlds = df['tld'].unique().size
tld_indicator = feature_column.indicator_column(tld_col)
feature_columns.append(tld_indicator)

# Prepare Provider column as a feature
provider_col = feature_column.categorical_column_with_vocabulary_list(key='provider',
                                                                      vocabulary_list=df['provider'].unique())
nb_providers = df['provider'].unique().size
provider_indicator = feature_column.indicator_column(provider_col)
feature_columns.append(provider_indicator)

# Prepare Fname column as a feature
fname_col = feature_column.categorical_column_with_vocabulary_list(key='fname', vocabulary_list=df['fname'].unique())
nb_fnames = df['fname'].unique().size
fname_indicator = feature_column.indicator_column(fname_col)
feature_columns.append(fname_indicator)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=20,
          verbose=1,
          callbacks=[tensorboard_callback]
          )
