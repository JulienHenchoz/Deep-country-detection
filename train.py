import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.python.keras.callbacks import RemoteMonitor

from classes.Loader import Loader
from classes.Utils import Utils
from tensorflow.keras import layers
from tensorflow import feature_column

df = Loader.load(
    './datasets/private/anonymized.csv',
    usecols=['tld', 'provider', 'fname', 'lname', 'target'],
)

df.dropna(inplace=True)
labels = LabelBinarizer().fit_transform(df['target'])
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
    feature_embedded = feature_column.embedding_column(feature_col, dimension=int(df[header].unique().size / 100))
    feature_columns.append(feature_embedded)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=50,
          verbose=1,
          )
