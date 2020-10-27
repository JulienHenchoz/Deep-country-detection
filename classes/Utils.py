import tensorflow as tf


class Utils:
    @staticmethod
    def dataframe_to_dataset(data_frame, target_column):
        data_frame = data_frame.copy()
        labels = data_frame[target_column]
        ds = tf.data.Dataset.from_tensor_slices((dict(data_frame), labels))
        ds = ds.shuffle(buffer_size=len(data_frame))
        return ds
