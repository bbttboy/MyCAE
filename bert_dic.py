import pandas as pd
import numpy as np

class MyBertDic():
    def __init__(self):
        self.data = pd.read_csv(r'D:\DataSet\fashion-200k\label_bert_feature.csv', header=None)
        self.dic = {}
        for i in range(len(self.data)):
            self.dic[self.data.iloc[i][0]] = self.data.iloc[i][1:].to_numpy(dtype='float32')

    def encode(self, text):
        features = []
        for t in text:
            features.append(self.dic[t])
        self.features_np = np.stack(features)
        return self.features_np
