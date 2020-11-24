"""class implements standard bidirectional LSTM recurrent layer"""
import torch
import torch.nn as nn
from src.layers.layer_birnn_base import LayerBiRNNBase
import torch.nn.functional as F

class LayerWCNN(LayerBiRNNBase):
    """BiLSTM layer implements standard bidirectional LSTM recurrent layer"""
    def __init__(self, args, input_dim, hidden_dim,cnn_layer,gpu):
        super(LayerWCNN, self).__init__(args, input_dim, hidden_dim, gpu)
        self.num_layers = 1
        self.num_directions = 2
        # rnn = nn.LSTM(input_size=input_dim,
        #               hidden_size=hidden_dim,
        #               num_layers=1,
        #               batch_first=True,
        #               bidirectional=True)
        # self.rnn = rnn
        self.args=args
        self.input_size = input_dim
        self.dropout = args.dropout_ratio
        self.gpu = gpu
        self.output_dim = 2*hidden_dim

        # self.word_feature_extractor == "CNN":
        # cnn_hidden = data.HP_hidden_dim
        self.word2cnn = nn.Linear(self.input_size, self.output_dim)
        self.cnn_layer = cnn_layer
        # print("CNN layer: ", self.cnn_layer)
        self.cnn_list = nn.ModuleList()
        self.cnn_drop_list = nn.ModuleList()
        self.cnn_batchnorm_list = nn.ModuleList()
        kernel = 3
        pad_size = int((kernel - 1) / 2)
        for idx in range(self.cnn_layer):
            self.cnn_list.append(
                nn.Conv1d(self.output_dim, self.output_dim, kernel_size=kernel, padding=pad_size))
            self.cnn_drop_list.append(nn.Dropout(self.dropout))
            self.cnn_batchnorm_list.append(nn.BatchNorm1d(self.output_dim))
        # The linear layer that maps from hidden state space to tag space

        # self.hidden2tag = nn.Linear(wcnn_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            # self.hidden2tag = self.hidden2tag.cuda()
            self.word2cnn = self.word2cnn.cuda()
            for idx in range(self.cnn_layer):
                self.cnn_list[idx] = self.cnn_list[idx].cuda()
                self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()

    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        # word_represent: Variable(batch_size, sent_len, hidden_dim)
        batch_size = input_tensor.size(0)
        input_tensor = input_tensor.cuda()
        word_in = torch.tanh(self.word2cnn(input_tensor)).transpose(2, 1).contiguous() #(bs, dim, max_seq_len) torch.Size([100, 200, 47])
        for idx in range(self.cnn_layer):
            if idx == 0:
                cnn_feature = F.relu(self.cnn_list[idx](word_in)) # torch.Size([100, 200, 47])
            else:
                cnn_feature = F.relu(self.cnn_list[idx](cnn_feature)) # torch.Size([100, 200, 47])
            cnn_feature = self.cnn_drop_list[idx](cnn_feature) # torch.Size([100, 200, 47])
            if batch_size > 1:
                cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature) # torch.Size([100, 200, 47])
        feature_out = cnn_feature.transpose(2, 1).contiguous() # torch.Size([100, 200, 47])

        return feature_out




        #
        # batch_size, max_seq_len, _ = input_tensor.shape
        # input_packed, reverse_sort_index = self.pack(input_tensor, mask_tensor)
        # h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        # c0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        # if self.args.if_elmo==False and self.args.if_bert==True and self.args.if_char==False and self.args.if_glove==False:
        #     input_packed =input_packed.cuda()
        # output_packed, _ = self.rnn(input_packed, (h0, c0))
        # output_tensor = self.unpack(output_packed, max_seq_len, reverse_sort_index)
        # return output_tensor  # shape: batch_size x max_seq_len x hidden_dim*2

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda
    #
    # def lstm_custom_init(self):
    #     nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
    #     nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse)
    #     nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
    #     nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse)
    #     self.rnn.bias_hh_l0.data.fill_(0)
    #     self.rnn.bias_hh_l0_reverse.data.fill_(0)
    #     self.rnn.bias_ih_l0.data.fill_(0)
    #     self.rnn.bias_ih_l0_reverse.data.fill_(0)
    #     # Init forget gates to 1
    #     for names in self.rnn._all_weights:
    #         for name in filter(lambda n: 'bias' in n, names):
    #             bias = getattr(self.rnn, name)
    #             n = bias.size(0)
    #             start, end = n // 4, n // 2
    #             bias.data[start:end].fill_(1.)
