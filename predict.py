import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python import feature_column
import sys
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from classes.Loader import Loader
from classes.Utils import Utils

df = Loader.load(
    './datasets/private/anonymized.csv',
    usecols=['tld', 'provider', 'fname', 'lname', 'target'],
)


model = tf.keras.models.load_model('281029')
fname = sys.argv[1]
provider = sys.argv[2].split('.')[0]
tld = sys.argv[2].split('.')[1]

df = pd.DataFrame(
    columns=['tld', 'provider', 'fname'],
    data=[
        [tld, provider, fname],
    ]
)

batch_size = 1
dataframe = df.copy()
ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
ds = ds.batch(batch_size)

labelEncoder = LabelEncoder()
labelEncoder.fit(df['target'])
classes = model.predict_classes(ds)
print('This guy is from ' + labelEncoder.inverse_transform(classes)[0])
