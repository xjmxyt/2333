# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ############# simple rnn model ####################### #
class TrajPreSimple(nn.Module):
    """baseline rnn model"""

    def __init__(self, num_locs, loc_emb_size, min_seq_len, tim_emb_size, hidden_size, use_cuda, rnn_type):
        super(TrajPreSimple, self).__init__()
        self.num_locs = num_locs
        self.loc_emb_size = loc_emb_size
        self.min_seq_len = min_seq_len
        self.tim_emb_size = tim_emb_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.rnn_type = rnn_type

        self.emb_loc = nn.Embedding(self.num_locs, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.min_seq_len, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.num_locs)
        self.dropout = nn.Dropout(p=0.3)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.to(self.device)
            c1 = c1.to(self.device)

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))
        out = out.squeeze(1)
        out = F.selu(out)
        out = self.dropout(out)

        y = self.fc(out)
        score = F.log_softmax(y)  # calculate loss by NLLoss
        return score


# ############# rnn model with attention ####################### #
class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size, device):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history, mask=False):
        '''
        history: [S1, B, N]
        out_state: [S2, B, N]
        '''
        seq_len = history.size()[0]
        state_len = out_state.size()[0]
        attn_energies = Variable(torch.zeros(state_len, seq_len)).to(self.device) # (S2, S1)
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.score(out_state[i], history[j])
        if mask:
            _negative = -1e11
            _negative_mask = torch.triu(torch.full_like(attn_energies, _negative), diagonal=1)
            attn_energies = torch.tril(attn_energies, diagonal=0)
            attn_energies = attn_energies + _negative_mask
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


# ##############long###########################
class TrajPreAttnAvgLongUser(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        self.num_locs = parameters.num_locs
        self.loc_emb_size = parameters.loc_emb_size
        self.min_seq_len = parameters.min_seq_len
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.use_cuda = parameters.use_cuda

        self.emb_loc = nn.Embedding(self.num_locs, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.min_seq_len, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.num_locs)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, history_loc, history_tim, history_count, uid, target_len):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.to(self.device)
            c1 = c1.to(self.device)

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        loc_emb_history = self.emb_loc(history_loc).squeeze(1)
        tim_emb_history = self.emb_tim(history_tim).squeeze(1)
        count = 0
        loc_emb_history2 = Variable(torch.zeros(len(history_count), loc_emb_history.size()[-1])).to(self.device)
        tim_emb_history2 = Variable(torch.zeros(len(history_count), tim_emb_history.size()[-1])).to(self.device)
        for i, c in enumerate(history_count):
            if c == 1:
                tmp = loc_emb_history[count].unsqueeze(0)
            else:
                tmp = torch.mean(loc_emb_history[count:count + c, :], dim=0, keepdim=True)
            loc_emb_history2[i, :] = tmp
            tim_emb_history2[i, :] = tim_emb_history[count, :].unsqueeze(0)
            count += c

        history = torch.cat((loc_emb_history2, tim_emb_history2), 1)
        history = F.tanh(self.fc_attn(history))

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out_state, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out_state, (h1, c1) = self.rnn(x, (h1, c1))
        out_state = out_state.squeeze(1)
        # out_state = F.selu(out_state)

        attn_weights = self.attn(out_state[-target_len:], history).unsqueeze(0)
        context = attn_weights.bmm(history.unsqueeze(0)).squeeze(0)
        out = torch.cat((out_state[-target_len:], context), 1)  # no need for fc_attn

        uid_emb = self.emb_uid(uid).repeat(target_len, 1)
        out = torch.cat((out, uid_emb), 1)
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y)

        return score


class TrajPreLocalAttnLong(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreLocalAttnLong, self).__init__()
        self.device = parameters.device
        self.num_locs = parameters.num_locs
        self.loc_emb_size = parameters.loc_emb_size
        self.min_seq_len = parameters.min_seq_len
        self.max_seq_len = 24
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.num_locs + 1, self.loc_emb_size) # include eos
        self.emb_tim = nn.Embedding(24, self.tim_emb_size) # test max t is 24

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size, self.device)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)  
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1)  # layer_num = 1
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)  
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.num_locs + 1)  # include eos
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            # nn.init.xavier_uniform(t)
            nn.init.xavier_uniform_(t)
        for t in hh:
            # nn.init.orthogonal(t)
            nn.init.orthogonal_(t)
        for t in b:
            # nn.init.constant(t, 0)
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, target_len):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        h2 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c2 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.to(self.device)
            h2 = h2.to(self.device)
            c1 = c1.to(self.device)
            c2 = c2.to(self.device)

        loc_emb = self.emb_loc(loc) # [[32],[3075],[32],[1564]]
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            # hidden_history, h1 = self.rnn_encoder(x[:-target_len], h1)
            hidden_state, h2 = self.rnn_decoder(x[-target_len:], h2)
            # hidden_history = torch.zeros_like(hidden_state)
            hidden_history = hidden_state
        elif self.rnn_type == 'LSTM':
            hidden_state, (h2, c2) = self.rnn_decoder(x[:target_len], (h2, c2))
            hidden_history = hidden_state
            
        hidden_history = hidden_history.squeeze(1) # [7, 1, 500] -> [7, 500]
        hidden_state = hidden_state.squeeze(1) # [7, 1, 500] -> [7, 500]

        attn_weights = self.attn(hidden_state, hidden_history, mask=True).unsqueeze(0) # [7, 7] -> [1, 7, 7]
        # Use a upper triangle zero mask to remove data from future trajectory.
        # lower_mask = torch.tril(torch.ones_like(attn_weights[0])).unsqueeze(0)
        # attn_weights = attn_weights * lower_mask
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0) # [7, 500]
        out = torch.cat((hidden_state, context), 1)  # no need for fc_attn  [7, 1000]
        out = self.dropout(out)

        y = self.fc_final(out) # [7, 3480]
        score = F.log_softmax(y)
        return score

    def inference(self, poi, EOS): # input: shape(1, ), [poi]
        # We use auto-regressive mechanism in the inference method.
        # We assume that x: (batch, input_size).
        t = torch.tensor([0]).to(self.device)
    
        with torch.no_grad():
            self.eval()
            h2 = Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device)
            c2 = Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device)
            path = []
            for i in range(24 - 1):
                loc_emb = self.emb_loc(poi).unsqueeze(0) 
                tim_emb = self.emb_tim(t).unsqueeze(0)
                x = torch.cat((loc_emb, tim_emb), 2)
                x = self.dropout(x)

                hidden_state, (h2, c2) = self.rnn_decoder(x, (h2, c2))
                hidden_state = hidden_state.squeeze(1) 
                hidden_history = hidden_state
                attn_weights = self.attn(hidden_state, hidden_history, mask=True).unsqueeze(0)
                context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
                out = torch.cat((hidden_state, context), 1) 
                out = self.dropout(out)
                y = self.fc_final(out) 
                score = F.log_softmax(y)
                poi = torch.argmax(score, dim = 1) # batch x locs -> batch
                if poi.item() == EOS:  
                    break
                t = t + 1
                path.append(poi.item()) 
            return path
        
        
class TrajPreLocalAttnLongTraffic(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, device, num_locs, loc_emb_size, tim_emb_size, hidden_size, 
                 attn_type, use_cuda, starting_dist, starting_sample, rnn_type='LSTM'):
        super(TrajPreLocalAttnLongTraffic, self).__init__()
        self.device = device
        self.num_locs = num_locs
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.hidden_size = hidden_size
        self.attn_type = attn_type
        self.use_cuda = use_cuda
        self.rnn_type = rnn_type
        self.starting_dist = starting_dist
        self.starting_sample = starting_sample
        if self.starting_sample == 'real':
            self.starting_dist = torch.tensor(starting_dist).float()

        self.emb_loc = nn.Embedding(self.num_locs + 1, self.loc_emb_size) # (batch_size,seq_len) -> (batch_size, seq_len, em_dim)
        self.emb_tim = nn.Embedding(48, self.tim_emb_size) # test max t is 24

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size, self.device)
        
        self.attn1 = nn.MultiheadAttention(self.hidden_size, 1)
        
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)  
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1)  # layer_num = 1
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)  
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.num_locs + 1)  # include eos
        self.dropout = nn.Dropout(p=0.3)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            # nn.init.xavier_uniform(t)
            nn.init.xavier_uniform_(t)
        for t in hh:
            # nn.init.orthogonal(t)
            nn.init.orthogonal_(t)
        for t in b:
            # nn.init.constant(t, 0)
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, target_len=48):
        h2 = Variable(torch.zeros(1, loc.shape[0], self.hidden_size)) # (1, B, H_out) num_layers=1
        c2 = Variable(torch.zeros(1, loc.shape[0], self.hidden_size)) # (1, B, H_cell)
        if self.use_cuda:
            h2 = h2.to(self.device)
            c2 = c2.to(self.device)

        loc_emb = self.emb_loc(loc) # (B, S, E1)
        tim_emb = self.emb_tim(tim) # (B, S, E2)
        x = torch.cat((loc_emb, tim_emb), 2) #(B, S, E)
        x = self.dropout(x)
        x = x.transpose(0, 1) # (S, B, E)
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_state, h2 = self.rnn_decoder(x, h2)
            hidden_history = hidden_state
        elif self.rnn_type == 'LSTM':
            print(h2.shape, c2.shape, x.shape)
            hidden_state, (h2, c2) = self.rnn_decoder(x, (h2, c2)) # (S, B, E)
            hidden_history = hidden_state
            print(h2.shape, c2.shape, x.shape)
            
        
        
        hidden_history = hidden_history.squeeze(1) # (S, B, E)
        hidden_state = hidden_state.squeeze(1) # (S, B, E)
        

        attn_weights = self.attn(hidden_state, hidden_history, mask=True).unsqueeze(0) # [7, 7] -> [1, 7, 7]
        # Use a upper triangle zero mask to remove data from future trajectory.
        # lower_mask = torch.tril(torch.ones_like(attn_weights[0])).unsqueeze(0)
        # attn_weights = attn_weights * lower_mask
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0) # [7, 500]
        out = torch.cat((hidden_state, context), 1)  # no need for fc_attn  [7, 1000]
        out = self.dropout(out)

        y = self.fc_final(out) # [7, 3480]
        score = F.log_softmax(y)
        return score
    
    def step(self, loc, tim, h2=None, c2=None):
        '''
        loc: starting loc
        tim: starting time
        h: lstm hidden state
        c: lstm cell state
        '''
        loc_emb = self.emb_loc(loc) # (32, 1, 1000)
        tim_emb = self.emb_tim(tim) # (32, 1, 48)
        # print(loc_emb.shape, tim_emb.shape)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            # hidden_history, h1 = self.rnn_encoder(x[:-target_len], h1)
            hidden_state, h2 = self.rnn_decoder(x, h2)
            # hidden_history = torch.zeros_like(hidden_state)
            hidden_history = hidden_state
        elif self.rnn_type == 'LSTM':
            hidden_state, (h2, c2) = self.rnn_decoder(x, (h2, c2))
            hidden_history = hidden_state
        hidden_history = hidden_history.squeeze(1) # [7, 1, 500] -> [7, 500]
        hidden_state = hidden_state.squeeze(1) # [7, 1, 500] -> [7, 500]

        attn_weights = self.attn(hidden_state, hidden_history, mask=True).unsqueeze(0)
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        out = torch.cat((hidden_state, context), 1) 
        out = self.dropout(out)
        y = self.fc_final(out) 
        score = F.softmax(y)
        out = score.multinomial(1)      
        return out, h2, c2
          
        
    def sample(self, batch_size, seq_len, x=None):
        """

        :param batch_size: int, size of a batch of training data
        :param seq_len: int, length of the sequence
        :param x: (batch_size, k), current generated sequence
        :return: (batch_size, seq_len), complete generated sequence
        """
        self.eval()
        res = []
        t = torch.tensor([0]).to(self.device)
        t = t.repeat(batch_size).reshape(batch_size, -1)
        flag = False  # whether sample from zero
        if x is None:
            flag = True
        s = 0
        if flag:
            if self.starting_sample == 'zero':
                x = Variable(torch.zeros((batch_size, 1)).long()).to(self.device)
            elif self.starting_sample == 'rand':
                x  = Variable(torch.randint(
                        high=self.total_locations, size=(batch_size, 1)).long()).to(self.device)
            elif self.starting_sample == 'real':
                x = Variable(torch.stack(
                    [torch.multinomial(self.starting_dist, 1) for i in range(batch_size)], dim=0)).to(self.device)
                s = 1
        h = Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device)
        c = Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device) 
        # print("x:", x.shape)
        # print(type(x))
        samples = [] 
        if flag:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                x, h, c = self.step(x, t, h, c)
                t = t + 1
                samples.append(x)
        else:
            raise ValueError("Not implemented")
        output = torch.cat(samples, dim=1)
        return output        
                
                
                              
            
                
    def inference(self, poi, EOS): # input: shape(1, ), [poi]
        # We use auto-regressive mechanism in the inference method.
        # We assume that x: (batch, input_size).
        t = torch.tensor([0]).to(self.device)
        with torch.no_grad():
            self.eval()
            h2 = Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device)
            c2 = Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device)
            if self.device:
                h2, c2 = h2.to(self.device), c2.to(self.device)
            path = []
            for i in range(24 - 1):
                loc_emb = self.emb_loc(poi).unsqueeze(0) 
                tim_emb = self.emb_tim(t).unsqueeze(0)
                x = torch.cat((loc_emb, tim_emb), 2)
                x = self.dropout(x)

                hidden_state, (h2, c2) = self.rnn_decoder(x, (h2, c2))
                hidden_state = hidden_state.squeeze(1) 
                hidden_history = hidden_state
                attn_weights = self.attn(hidden_state, hidden_history, mask=True).unsqueeze(0)
                context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
                out = torch.cat((hidden_state, context), 1) 
                out = self.dropout(out)
                y = self.fc_final(out) 
                score = F.log_softmax(y)
                poi = torch.argmax(score, dim = 1) # batch x locs -> batch
                if poi.item() == EOS:  
                    break
                t = t + 1
                path.append(poi.item()) 
            return path