import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class CNN(nn.Module):
    def __init__(self,config, n_classes):
        super(CNN,self).__init__()
        self.bert  = BertModel.from_pretrained(config['transformerModel'])

        self.convs = nn.ModuleList([nn.Conv2d(1,config['n_kernels'],(ker,768)) for ker in config['kernels']])
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(in_features=len(config['kernels']) * config['n_kernels'], out_features=n_classes)

        if config['freez_transformer']:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self,input_ids, attention_masks):
        out = self.bert(input_ids= input_ids, attention_mask = attention_masks)
        last_hidden_state_cls = out[0]
        x = last_hidden_state_cls.unsqueeze(1)
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class LSTM(nn.Module):
    def __init__(self,config,n_classes):
        super(LSTM, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['transformerModel'])
        self.x = 2 if config['bidirectional'] else 1

        self.lstm = nn.LSTM(input_size=768,hidden_size=config['hidden_size'],
                            num_layers=config['n_lstm_layers'],
                            bidirectional=config['bidirectional'], batch_first=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(in_features=(config['hidden_size'] * self.x), out_features=n_classes)

        if config['freez_transformer']:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_masks):
        out = self.bert(input_ids =input_ids, attention_mask = attention_masks)
        last_hidden_state_cls = out[0]
        h = torch.zeros((self.config['n_lstm_layers'] * self.x, self.config['batch_size'], self.config['hidden_size']))
        c = torch.zeros((self.config['n_lstm_layers'] * self.x, self.config['batch_size'], self.config['hidden_size']))
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out, (hidden, cell) = self.lstm(last_hidden_state_cls, (h, c))
        logits = self.fc(out[:, -1])
        return logits

