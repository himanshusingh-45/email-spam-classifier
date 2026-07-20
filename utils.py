import os
import pandas as pd
import requests
from io import BytesIO
import zipfile

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        return DATA_FILE
    r = requests.get(DATA_URL, timeout=30)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall(DATA_DIR)
    return DATA_FILE

def load_sms_spam():
    path = download_dataset()
    df = pd.read_csv(path, sep='\t', header=None, names=['label','text'])
    df['label'] = df['label'].map({'ham':0,'spam':1})
    return df
