import io

import PIL
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

import lib.datasets.lmdb.caffe_pb2 as caffe_pb2

class CaffeLMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        
        self.keys = self.get_keys()

    def open_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
        self.txn = self.env.begin()
    
    def close_db(self):
        self.env.close()
        self.txn = None
        self.env = None 

    def __len__(self):
        return len(self.keys)

    def get_keys(self):
        keys = None
        self.open_db()
        with self.env.begin() as txn:
            keys = [None]*txn.stat()['entries']
            cursor = txn.cursor()
            i = 0
            while cursor.next():
                keys[i] = cursor.key()
                i += 1
        self.close_db()
        return keys
                

    def __getitem__(self, index):
        if self.env is None:
            self.open_db()

        key = self.keys[index]
        value = self.txn.get(key)
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = datum.label
        # Process the value as per your requirements
        # Here, we assume the value is a serialized tensor
        im_pil = PIL.Image.open(io.BytesIO(datum.data))

        if self.transform:
            trans_im = self.transform(im_pil)
        else:
            trans_im = torch.tensor(np.array(im_pil, dtype=np.float32)/255.0)
        return trans_im, label