import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self,config, embedding, n_classes):
        super(CNN,self).__init__()
        self.config = config
        self.embed_dim = embedding.size()[1]
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze = self.config['freeze_embd'])
        self.n_classes = n_classes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=self.config['n_kernels'],
                      kernel_size=self.config['kernels'][i])
            for i in range(len(self.config['kernels']))
        ])
        self.dropout = nn.Dropout(self.config['dropout'])
        self.fc = nn.Linear(in_features=len(self.config['kernels']) * self.config['n_kernels'], out_features= self.n_classes)
    def forward(self, sentences):
        # sentences: batch * maxlen
        x = self.embedding(sentences.long())
        x = self.dropout(x)
        # x: batch * maxlen * embed_dim
        x = x.permute(0,2,1)
        #x: batch * embed_dim * maxlen
        x = x.to(torch.float32)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x]
        x = torch.cat([x_pool.squeeze(dim=2) for x_pool in x], dim=1)
        logits = self.fc(self.dropout(x))
        return logits


class LSTM(nn.Module):
    def __init__(self, config, embedding, n_classes):

        super(LSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze = self.config['freeze_embd'])
        self.embed_dim = embedding.size()[1]
        self.n_classe = n_classes
        self.dropout = nn.Dropout(config['dropout'])
        self.bidirectional = config['bidirectional']
        self.hidden_size = config['hidden_size']

        self.x = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(self.embed_dim,self.hidden_size,
                            num_layers=self.config['n_lstm_layers'],
                            bidirectional= self.config['bidirectional'],
                            batch_first= True)
        self.fc = nn.Linear(self.hidden_size * self.x, self.n_classe)

    def attention(self, rnn_out, state):
        # print('rnn_out, state',rnn_out.size(), state.size())
        merged_state = torch.cat([s for s in state[-2:]], 1)
        # print('merged_state',merged_state.size())
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # print('merged_state', merged_state.size())
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        # print('weights',weights.size())
        weights = torch.nn.functional.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        # print('weights', weights.size())
        # print('torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)',torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2).size())
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)
    def forward(self, sentences):
        input = self.embedding(sentences.long())
        input = self.dropout(input)
        h = torch.zeros((self.config['n_lstm_layers'] * self.x, sentences.size(0), self.config['hidden_size']))
        c = torch.zeros((self.config['n_lstm_layers'] * self.x, sentences.size(0), self.config['hidden_size']))

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        input = input.to(torch.float32)
        out, (hidden,cell) = self.lstm(input, (h, c))
        if self.config['attention']:
            logits = self.fc(self.attention(out, hidden))
        else:
            logits = self.fc(out[:,-1])
        return logits