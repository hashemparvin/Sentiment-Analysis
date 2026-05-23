import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from utils import clean_text
from dataset import TweetDataset, collate
from config import MAX_LEN

def load_raw_data(path="Tweets.csv"):
    df = pd.read_csv(path)

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    df = df[["text","airline_sentiment"]]
    df["label"] = df["airline_sentiment"].map(label_map)

    df["text"] = df["text"].apply(clean_text)

    return df


def split_and_balance(df):
    train_text, temp_text, train_y, temp_y = train_test_split(
        df["text"],
        df["label"],
        test_size=0.30,
        stratify=df["label"],
        random_state=42
    )

    val_text, test_text, val_y, test_y = train_test_split(
        temp_text,
        temp_y,
        test_size=0.50,
        stratify=temp_y,
        random_state=42
    )

    ros = RandomOverSampler(random_state=42)

    train_text_np = np.array(train_text).reshape(-1,1)
    train_y_np = np.array(train_y)

    train_text_res, train_y_res = ros.fit_resample(train_text_np,train_y_np)

    train_text = train_text_res.flatten()
    train_y = train_y_res

    return (train_text,train_y), (val_text,val_y), (test_text,test_y)


def get_dataloaders(batch_size=8, path="Tweets.csv"):
    df = load_raw_data(path)

    (train_text,train_y), (val_text,val_y), (test_text,test_y) = split_and_balance(df)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    train_ds = TweetDataset(train_text, train_y, tokenizer)
    val_ds   = TweetDataset(list(val_text), list(val_y), tokenizer)
    test_ds  = TweetDataset(list(test_text), list(test_y), tokenizer)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate)

    val_loader   = DataLoader(val_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate)

    test_loader  = DataLoader(test_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate)

    return train_loader, val_loader, test_loader
