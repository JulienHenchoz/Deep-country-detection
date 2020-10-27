import numpy as np


class Sanitizer:
    @staticmethod
    def remove_accents(df):
        cols = df.select_dtypes(include=[np.object]).columns
        df[cols] = df[cols].apply(lambda x: x.str
                                  .normalize('NFKD')
                                  .str
                                  .encode('ascii', errors='ignore')
                                  .str
                                  .decode('utf-8')
                                  .astype(str))
        return df

