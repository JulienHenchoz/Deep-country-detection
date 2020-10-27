import pandas as pd

from classes.Sanitizer import Sanitizer


class Loader:
    @staticmethod
    def load(file_name, usecols=()):
        separator = ','
        df = pd.read_csv(file_name, sep=separator, usecols=usecols)

        # Remove all accents from columns and force all columns to be interpreted as strings
        df = Sanitizer.remove_accents(df)
        return df
