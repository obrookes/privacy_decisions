import os
import pandas as pd

import json
from glob import glob

class DatasetLoader:

    def __init__(self):
        self.dataset = None
        self.attr = None
        self.file_paths=None
        self.labels=[]

    def flatten_label(self,index,file_name):
        with open(file_name) as json_file:
            encoding = int(1)
            data = json.load(json_file)
            data.update(dict.fromkeys(data['labels'],encoding))
            data.pop('labels')
        return pd.DataFrame(data,index=[index])


    def load_data(self,file_paths):
        self.get_file_paths(file_paths)
        self.create_df()
        self.set_attr()
        self.encode_dataset()
        self.get_label_encodings()
        self.insert_labels()
        return self.dataset

    def create_df(self):
        self.dataset = pd.concat(
                [self.flatten_label(counter,file) for counter, file in enumerate(self.file_paths)]
                )

    def get_file_paths(self,file_paths):
        ext = '*.json'
        self.file_paths=(glob(os.path.join(file_paths,ext)))

    def encode_dataset(self):
        attr = self.get_attr()
        self.dataset[attr]=self.dataset[attr].fillna(0)

    def get_dataset(self):
        self.encode_dataset()
        return self.dataset

    # Getting & setting attributes

    def set_attr(self):
        self.attr = self.dataset.columns[4:]

    def get_attr(self):
        return self.attr

    # Getting label encodings 

    def get_label_encodings(self):
        attr = self.get_attr()
        for index, row in self.dataset[attr].iterrows():
            self.labels.append(row.to_numpy())

    def insert_labels(self):
        self.dataset.insert(loc=len(self.dataset.columns),
                            column='encoded_label',
                            value=self.labels)

