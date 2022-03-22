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
        # TODO is this actually the last in the time sequence???
        # need to the check the input against the known dataset sequence
        # just use the last element in the output sequence 
        return output[:, -1, :]
