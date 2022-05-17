import torch.nn as nn
import torch

class ScrewingModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ScrewingModel, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_T):
        input = input_T.float()
        lstm_out, _ = self.lstm(input)
        # print(lstm_out.size())
        output_T = self.hidden2out(lstm_out)
        # print(output_T)
        # print(output_T[])

        output = output_T.float()
        
        # just use the last element in the output sequence 
        return output[:, -1, :]

class ScrewingModelSeq(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ScrewingModelSeq, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_T):
        input = input_T.float()
        lstm_out, _ = self.lstm(input)
        # print(lstm_out.size())
        output_T = self.hidden2out(lstm_out)
        # print(output_T)
        # print(output_T[])

        output = output_T.float()
        
        # just use the last element in the output sequence 
        # return output[:, -1, :]
        return output


class ScrewingModelGaussian(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, full_seq=False):
        super(ScrewingModelGaussian, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, 2*output_dim)
        
        self.full_seq = full_seq

    def forward(self, input_T):
        input = input_T.float()
        lstm_out, _ = self.lstm(input)
        # print(lstm_out.size())
        output_T = self.hidden2out(lstm_out)
        # print(output_T)
        # print(output_T[])

        output = output_T.float()
        
        # exponentiate the log var part
        output[:, :, self.output_dim:] = torch.exp(output[:, :, self.output_dim:])
        
        # just use the last element in the output sequence 
        if self.full_seq:
            return output #Batch, Time window, Output dim
        else:
            return output[:, -1, :] #Batch, Time window, Output dim

