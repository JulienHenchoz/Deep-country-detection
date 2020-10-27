import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

from classes.Encoder import Encoder
from classes.Loader import Loader

df = Loader.load(
    './datasets/private/anonymized.csv',
    usecols=['tld', 'provider', 'First Name', 'Last Name', 'Country']
)

encoder = Encoder()
"""
Fit on all possible values 
"""
print("Fitting encoder for all data types...")
encoder.fit(df, 'tld')
encoder.fit(df, 'provider')
#encoder.fit(df, 'First Name')
#encoder.fit(df, 'Last Name')
encoder.fit(df, 'Country')

"""
Train on a slice of data only
"""

print("Binarizing the dataset...")
binarized_tlds = encoder.encode(df, 'tld').tolist()
binarized_providers = encoder.encode(df, 'provider').tolist()
#binarized_first_names = encoder.encode(df, 'First Name').tolist()
#binarized_last_names = encoder.encode(df, 'Last Name').tolist()
binarized_countries = encoder.encode(df, 'Country').tolist()

y = np.array(binarized_countries)
X = np.concatenate((binarized_tlds, binarized_providers), axis=1)

"""
  Create model and train it with our encoded data 
"""
model = tf.keras.Sequential([
  tf.keras.layers.Dense(X.shape[1], activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(y.shape[1])
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.fit(X, y, epochs=5)

model.save('./models/251020')