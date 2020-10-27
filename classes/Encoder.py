from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import tensorflow as tf
import numpy as np

class Encoder:
    def __init__(self):
        self.label_encoders = {}
        self.binarizers = {}

    def fit(self, df, column_name):
        all_values = df[column_name].str.lower().unique()
        label_encoder = LabelEncoder()
        possible_values_as_integers = label_encoder.fit_transform(all_values)
        binarizer = LabelBinarizer()
        binarizer.fit(possible_values_as_integers)
        self.label_encoders[column_name] = label_encoder
        self.binarizers[column_name] = binarizer

    def encode(self, df, column_name):
        label_encoder = self.label_encoders[column_name]
        binarizer = self.binarizers[column_name]
        dict_data = df.to_dict(orient='records')
        values_as_integers = label_encoder.transform([row[column_name].lower() for row in dict_data])
        return binarizer.transform(values_as_integers)

    def decode_binary(self, values, column_name):
        values_as_integer = self.binarizers[column_name].inverse_transform(values)
        return self.label_encoders[column_name].inverse_transform(values_as_integer)

    def decode_integer(self, values, column_name):
        return self.label_encoders[column_name].inverse_transform(values)
