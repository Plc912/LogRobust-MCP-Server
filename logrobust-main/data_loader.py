from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle


class AliyunDataLoaderForFed(Dataset):
    def __init__(self,mode='train',semi=False,rank=1,world_size=3,num_keys=219) -> None:
        super().__init__()
        self.num_keys = num_keys
        x=np.load("../Fedlog/data/data_{}.npy".format(rank-1))
        if semi:
            y=np.load("../Fedlog/data/semi_label_{}.npy".format(rank-1))
        else:
            y=np.load("../Fedlog/data/label_{}.npy".format(rank-1))
        y[y!=0] = 1
        _len = len(y)
        if mode == 'train':
            self.x = x[:int(_len*0.8)]
            self.y = y[:int(_len*0.8)]
        else:
            self.x = x[int(_len*0.8):]
            self.y = y[int(_len*0.8):]
        with open("EventId2WordVecter.pickle",'br') as f:
            self.EventId2WordVecter = pickle.load(f)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        eventId_list = self.x[index]
        x=[]
        for eventId in eventId_list:
            x.append(self.EventId2WordVecter[eventId])
        return np.array(x).astype(np.float32),self.y[index]

class AliyunDataLoader(Dataset):
    def __init__(self,mode='train') -> None:
        super().__init__()

        x=np.load("data/data.npy")
        y=np.load("data/label.npy")
        with open("EventId2WordVecter.pickle",'br') as f:
            self.EventId2WordVecter = pickle.load(f)
        y[y!=0] = 1
        _len = len(y)
        if mode == 'train':
            self.x = x[:int(_len*0.8)]
            self.y = y[:int(_len*0.8)]
        else:
            self.x = x[int(_len*0.8):]
            self.y = y[int(_len*0.8):]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        eventId_list = self.x[index]
        x=[]
        for eventId in eventId_list:
            x.append(self.EventId2WordVecter[eventId])
        return np.array(x).astype(np.float32),self.y[index]


class CMCCDataLoaderForFed(Dataset):
    def __init__(self,mode='train',semi=False,rank=1,world_size=3,num_keys=144) -> None:
        super().__init__()
        self.num_keys = num_keys
        data_path = "/home/zhangshenglin/chezeyu/log/cmcc_0929/data"
        x=np.load("{}/eventIndex_{}.npy".format(data_path,rank-1))
        if semi:
            y=np.load("{}/semi_label_{}.npy".format(data_path,rank-1))
        else:
            y=np.load("{}/label_{}.npy".format(data_path,rank-1))
        y[y!=0] = 1
        _len = len(y)
        if mode == 'train':
            self.x = x[:int(_len*0.8)]
            self.y = y[:int(_len*0.8)]
        else:
            self.x = x[int(_len*0.8):]
            self.y = y[int(_len*0.8):]
        with open("EventId2WordVecter_cmcc.pickle",'br') as f:
            self.EventId2WordVecter = pickle.load(f)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        eventId_list = self.x[index]
        x=[]
        for eventId in eventId_list:
            x.append(self.EventId2WordVecter[eventId])
        return np.array(x).astype(np.float32),self.y[index]