import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd


class IEMOCAPDataset(Dataset):
    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('data/iemocap_multimodal_features.pkl', 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('E:\Code\SDT-main\data\iemocap_features_comet.pkl', 'rb'),
                          encoding='latin1')

        self.len = len(self.keys)



    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<13 else pad_sequence(dat[i], True) if i<15 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('E:\Code\SDT-main\data\meld_features_comet.pkl', 'rb'),
                          encoding='latin1')

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<13 else pad_sequence(dat[i], True) if i<15 else dat[i].tolist() for i in dat]