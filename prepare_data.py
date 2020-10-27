import pandas as pd
import tensorflow as tf
from classes import Utils

file_name = './datasets/private/mailchimp_subscribed.csv'
separator = ','
df = pd.read_csv(file_name,
                 sep=separator,
                 usecols=['Email Address', 'First Name', 'Last Name', 'Country']
 )


splitted_emails = df['Email Address'].str.split('@', expand=True)[1]
tlds = splitted_emails.apply(lambda x: x.split('.')[-1]).str.lower()
providers = splitted_emails.apply(lambda x: x.split('.')[0]).str.lower()
df['provider'] = providers
df['tld'] = tlds

df = df.drop(columns='Email Address')

df.to_csv('anonymized.csv')