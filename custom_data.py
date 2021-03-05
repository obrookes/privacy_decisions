import os
import torch
from torch.utils.data import Dataset
from skimage import io

class PrivacyDataset(Dataset):

    def __init__(self,path,loaded_data,transform):
        self.img_ids=loaded_data['id']
        self.labels=loaded_data['encoded_label']
        self.transform=transform
        self.path=path

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        try:
            ext='.jpg'
            img_id=self.img_ids.iloc[index]+ext
            img_path=os.path.join(self.path,img_id)
            img=io.imread(img_path)
        except:
            ext='.png'
            img_id=self.img_ids.iloc[index]+ext
            img_path=os.path.join(self.path,img_id)
            img=io.imread(img_path)
        
        if self.transform:
            img = self.transform(img)

        label=torch.Tensor(self.labels.iloc[index])
        return(img,label)


         
