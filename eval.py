import numpy as np
import tensorflow as tf

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
encoder.fit(df, 'First Name')
encoder.fit(df, 'Last Name')
encoder.fit(df, 'Country')

"""
Eval on a slice of data only
"""
df = df[1000:2000]

print("Binarizing the dataset...")
binarized_tlds = encoder.encode(df, 'tld').tolist()
binarized_providers = encoder.encode(df, 'provider').tolist()
binarized_first_names = encoder.encode(df, 'First Name').tolist()
binarized_last_names = encoder.encode(df, 'Last Name').tolist()
binarized_countries = encoder.encode(df, 'Country').tolist()


X = np.concatenate((binarized_tlds, binarized_providers, binarized_first_names), axis=1)
y = np.array(binarized_countries)


print("Loading model...")
model = tf.keras.models.load_model('./models/251020')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X, y, batch_size=128)
print("test loss, test acc:", results)
