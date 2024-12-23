import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing

import numpy as np
import json

import os

from torch.utils.data import Dataset, DataLoader

with open("arxivData.json", 'r') as f:
    data = json.load(f)

class ArxivDataset(Dataset):
    
    def __init__(self, fpath, transform=None):
        super().__init__()
        self.transform = transform
        with open(fpath, 'r') as f:
            self.data = json.load(f)
        self.id2idx = preprocessing.LabelEncoder()
        self.id2idx.fit_transform([self.data[idx]['id'] for idx in range(len(self.data))])
    
    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        arxiv_id = item['id'] 
        author = item['author']
        year, month, day = item['year'], item['month'], item['day']
        link = item['link']
        summary = item['summary']
        tag = item['tag']

        if self.transform:
            inputs = self.transform(summary)
        else:
            inputs = summary

        return inputs, arxiv_id, author, year, month, day, link, tag
    

dataset = ArxivDataset(fpath="arxivData.json")
data_loader = DataLoader(dataset, batch_size=16)

model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')

embeddings = model.encode(next(iter(data_loader))[0][0])

nth_b = 0
embed_list = []
for b in data_loader:
    
    nth_b += 1
    embed_list.append(model.encode(b[0]))

    if nth_b % 500 == 0:
        print(f"current batch number = {nth_b}")
        break

xb = np.concatenate(embed_list)

d = embeddings.shape[0]
nb = xb.shape[0]
nq = nb // 100
nlists = 100
metric = faiss.METRIC_INNER_PRODUCT

faiss_index = faiss.IndexFlatIP(d)
faiss_index.add(xb)
print(faiss_index.ntotal)
k = 4

D, I = faiss_index.search(xb[3:4], k)

print(D)
print(I)