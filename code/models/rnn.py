import torch 
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class RNN(nn.Module):
    def __init__(self, 
                 device=None,
                 rnn_type='LSTM',
                 total_locations=8606,
                 embedding_net=None,
                 embedding_dim=32, 
                 hidden_dim=64,
                 num_layers=2,
                 bidirectional=False,    
                 starting_sample = "zero",
                 starting_dist=None
                ):
        super(RNN, self).__init__()
        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
        
        self.device = device 
        self.total_locations = total_locations 
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        if embedding_net:
            self.embedding = embedding_net
        else:
            self.embedding = nn.Embedding(
                num_embeddings=total_locations, embedding_dim=embedding_dim)
            
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                                bidirectional=bidirectional, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                                bidirectional=bidirectional, batch_first=True)  
                      
        self.linear = nn.Linear(hidden_dim, total_locations)
        
        self.starting_sample = starting_sample
        
        if self.starting_sample == 'real':
            self.starting_dist = torch.tensor(starting_dist).float()
            
        self.linear_dim = hidden_dim * 2 if bidirectional else hidden_dim
                
    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)        

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(
            (2*self.num_layers if self.bidirectional else self.num_layers, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros(
            (2*self.num_layers if self.bidirectional else self.num_layers, batch_size, self.hidden_dim)))
        if self.device:
            h, c = h.to(self.device), c.to(self.device)
        return h, c

    def forward(self, x):
        """

        :param x: (batch_size, seq_len), sequence of locations
        :return:
            (batch_size * seq_len, total_locations), prediction of next stage of all locations
        """
        self.rnn.flatten_parameters()
        x = self.embedding(x)
        h0, c0 = self.init_hidden(x.size(0))
        x, (h, c) = self.rnn(x, (h0, c0))
        pred = F.log_softmax(self.linear(
            x.contiguous().view(-1, self.linear_dim)), dim=-1)
        return pred
    
    def step(self, x):
        """

        :param x: (batch_size, seq_len), sequence of locations
        :return:
            (batch_size * seq_len, total_locations), prediction of next stage of all locations
        """
        self.rnn.flatten_parameters()
        x = self.embedding(x)
        h0, c0 = self.init_hidden(x.size(0))
        x, (h, c) = self.rnn(x, (h0, c0))
        pred = F.softmax(self.linear(
            x.contiguous().view(-1, self.linear_dim)), dim=-1)
        return pred
            
    def sample(self, batch_size, seq_len, x=None):
        """

        :param batch_size: int, size of a batch of training data
        :param seq_len: int, length of the sequence
        :param x: (batch_size, k), current generated sequence
        :return: (batch_size, seq_len), complete generated sequence
        """
        res = []
        flag = False  # whether sample from zero
                
        #self.attn.flatten_parameters()

        if x is None:
            flag = True
        s = 0
        if flag:
            if self.starting_sample == 'zero':
                x = torch.LongTensor(torch.zeros((batch_size, 1))).to(self.device)
            elif self.starting_sample == 'rand':
                x  = torch.LongTensor(torch.randint(
                        high=self.total_locations, size=(batch_size, 1))).to(self.device)
            elif self.starting_sample == 'real':
                x = torch.LongTensor(torch.stack(
                    [torch.multinomial(self.starting_dist, 1) for i in range(batch_size)], dim=0)).to(self.device)
                s = 1

        if self.device:
            x = x.to(self.device)
        samples = []
        if flag:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                x = self.step(x)
                # np.save('x.npy',x.detach().numpy())
                x = x.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                x = self.step(lis[i])
                samples.append(lis[i])
            x = x.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)                                
                x = self.step(x)
                x = x.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output     
        
        