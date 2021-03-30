import pickle
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from transformers import BertTokenizer
class deepPreProcessing():
    def __init__(self,config):
        self.config = config
        self.data,self.n_classes = self.read(self.config['path_to_train_data'])
        self.max_len = max(len(a) for a in self.data.tokenized_text)
        self.Word2index = self.createWord2index()
        self.train, self.test, self.valid = self.splitData()
        self.train['wordIds'] = self.encode(self.train)
        self.test['wordIds'] = self.encode(self.test)
        self.valid['wordIds'] = self.encode(self.valid)
        self.embeddingWeights = self.load_pretrained_WE()


    def read(self, path_to_file):
        df = pd.read_csv(path_to_file, sep='\t', header=None, names=['text', 'label'])
        n_classes = len(set(df.label.values))
        df.text = [self.textPreprocessing_dl(s) for s in df.text.values]
        df['tokenized_text'] = [self.tokenize(s) for s in df.text.values]
        labels = list(set(df.label.values))
        dict = {}
        for i,l in enumerate(labels):
            dict[l] = i
        df.label.replace(dict, inplace=True)
        return df, n_classes

    def splitData(self):
        if self.config['path_to_test_data']!=None:
            test = self.read(self.config['path_to_test_data'])
        else:
            train, test = train_test_split(self.data, test_size=0.3,
                                                     stratify=self.data.label,
                                                     random_state=42)
        train, valid = train_test_split(train, test_size=0.2,
                                                      stratify=train.label,
                                                      random_state=42)
        train.reset_index(drop=True,inplace=True)
        test.reset_index(drop=True,inplace=True)
        valid.reset_index(drop=True,inplace=True)
        return train, test, valid

    def tokenize(self,s):
        return word_tokenize(s)

    def textPreprocessing_dl(self,s):
        """
            - Lowercase the sentence
            - Change "'t" to "not"
            - Remove "@name"
            - Isolate and remove punctuations except "?"
            - Remove other special characters
            - Remove stop words except "not" and "can"
            - Remove trailing whitespace
            """
        stop_words = set(stopwords.words('english'))
        s = s.lower()
        # Change 't to 'not'
        s = re.sub(r"\'t", " not", s)
        # Remove @name
        s = re.sub(r'(@.*?)[\s]', ' ', s)
        # Isolate and remove punctuations except '?'
        s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
        s = re.sub(r'[^\w\s\?]', ' ', s)
        # Remove some special characters
        s = re.sub(r'([\;\:\|•«\n])', ' ', s)
        # Remove stopwords except 'not' and 'can'
        s = " ".join([word for word in s.split()
                      if word not in stop_words
                      or word in ['not', 'can']])
        # Remove trailing whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    def dataLoader(self,df,shuffle=True):
        x = list([list(a) for a in df.wordIds.values])
        samples, lables = tuple(torch.tensor(data) for data in (x, df.label.values))
        data = TensorDataset(samples, lables)

        Sampler = RandomSampler(data)
        dataLoader = DataLoader(data, shuffle=shuffle,sampler=None,
                                batch_size=self.config['batch_size'],
                                drop_last=False)
        return dataLoader
    def createWord2index(self):
        if self.config['path_to_pretrained_WE'] == None:
            word2idx = {}
            word2idx['<pad>'] = 0
            word2idx['<unk>'] = 1
            idx = 2
            for tokenized_sent in self.data.tokenized_text.values:

                for token in tokenized_sent:
                    if token not in word2idx:
                        word2idx[token] = idx
                        idx += 1
        else:
            with open(self.config['path_to_pretrained_WE'],'r') as f:
                word2idx = {}
                word2idx['<pad>'] = 0
                word2idx['<unk>'] = 1
                idx = 2
                for line in f:
                    record = line.strip().split()
                    token = record[0]
                    if token not in word2idx:
                        word2idx[token] = idx
                        idx += 1


        return word2idx

    def encode(self, df):
        input_ids = []
        for tokenized_sent in df.tokenized_text.values:
            tokenized_sent += ['<pad>'] * (self.max_len - len(tokenized_sent))
            input_id = []
            for token in tokenized_sent:
                if token in self.Word2index:
                    input_id.append(self.Word2index[token])
                else:
                    input_id.append(self.Word2index['<unk>'])
            # input_id = [self.Word2index[token] for token in tokenized_sent][:self.max_len]
            input_ids.append(np.array(input_id,dtype=np.int64))
        return torch.Tensor(np.array(input_ids,dtype=np.int64))

    def load_pretrained_WE(self):
        if self.config['path_to_pretrained_WE'] == None:
            print('There is no pretrained wordd embedding model!!')
            embeddings = np.random.uniform(-0.25, 0.25, (len(self.Word2index), self.config['embd_dim']))

        else:
            fin = open(self.config['path_to_pretrained_WE'], 'r',
                       encoding='utf-8', newline='\n', errors='ignore')
            n, d = map(int, fin.readline().split())
            embeddings = np.random.uniform(-0.25,0.25,(len(self.Word2index),d))
            embeddings[self.Word2index['<pad>']] = np.zeros((d,))
            count = 0
            for line in fin:
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                if word in self.Word2index:
                    count += 1
                    embeddings[self.Word2index[word]] = np.array(tokens[1:],dtype=np.float32)
        return torch.tensor(embeddings, dtype=torch.double)
############################################################################
class transPreProcessing():
    def __init__(self,config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['transformerModel'],do_lower_case = True)
        self.data, self.n_classes = self.read(config['path_to_train_data'])
        self.train, self.test, self.valid = self.splitData()
        self.maxlen = max(len(x.split()) for x in self.data.text.values)

    def encode(self,df):
        tokens = self.tokenizer.batch_encode_plus(
            df.text.tolist(),
            max_length = self.maxlen,
            add_special_tokens= True,
            padding = True,
            truncation= True,
            return_token_type_ids= False
        )
        return tokens
    def dataLoader(self, df, shuffle=True):
        tokens = self.encode(df)
        data_seq = torch.tensor(tokens['input_ids'])
        data_mask = torch.tensor(tokens['attention_mask'])
        data_y = torch.tensor(df.label.tolist())

        data = TensorDataset(data_seq, data_mask, data_y)
        data_sampler = RandomSampler(data)
        data_loader = DataLoader(data, sampler=None, batch_size= self.config['batch_size'],
                                 shuffle=shuffle, drop_last=True)

        return data_loader

    def read(self, path_to_file):
        df = pd.read_csv(path_to_file, sep='\t', header=None, names=['text', 'label'])
        n_classes = len(set(df.label.values))
        df.text = [self.textPreprocessing_dl(s) for s in df.text.values]

        labels = list(set(df.label.values))
        dict = {}
        for i,l in enumerate(labels):
            dict[l] = i
        df.label.replace(dict, inplace=True)
        return df, n_classes

    def splitData(self):
        if self.config['path_to_test_data']!=None:
            test = self.read(self.config['path_to_test_data'])
        else:
            train, test = train_test_split(self.data, test_size=0.3,
                                                     stratify=self.data.label,
                                                     random_state=42)
        train, valid = train_test_split(train, test_size=0.2,
                                                      stratify=train.label,
                                                      random_state=42)
        train.reset_index(drop=True,inplace=True)
        test.reset_index(drop=True,inplace=True)
        valid.reset_index(drop=True,inplace=True)
        return train, test, valid


    def textPreprocessing_dl(self,s):
        """
            - Lowercase the sentence
            - Change "'t" to "not"
            - Remove "@name"
            - Isolate and remove punctuations except "?"
            - Remove other special characters
            - Remove stop words except "not" and "can"
            - Remove trailing whitespace
            """
        stop_words = set(stopwords.words('english'))
        s = s.lower()
        # Change 't to 'not'
        s = re.sub(r"\'t", " not", s)
        # Remove @name
        s = re.sub(r'(@.*?)[\s]', ' ', s)
        # Isolate and remove punctuations except '?'
        s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
        s = re.sub(r'[^\w\s\?]', ' ', s)
        # Remove some special characters
        s = re.sub(r'([\;\:\|•«\n])', ' ', s)
        # Remove stopwords except 'not' and 'can'
        s = " ".join([word for word in s.split()
                      if word not in stop_words
                      or word in ['not', 'can']])
        # Remove trailing whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s

###########################################################################
class tradPreProcessing():
    def __init__(self,config):
        self.config = config
    def read(self):
        pass
