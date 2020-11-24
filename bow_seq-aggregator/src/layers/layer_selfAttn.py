"""class implements standard bidirectional LSTM recurrent layer"""
import torch
import torch.nn as nn
from src.layers.layer_birnn_base import LayerBiRNNBase
import torch.nn.functional as F
from src.layers.layer_bilstm import LayerBiLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class LayerSelfAttn(LayerBiRNNBase):
    """BiLSTM layer implements standard bidirectional LSTM recurrent layer"""
    def __init__(self, args, input_dim, hidden_dim, gpu):
        super(LayerSelfAttn, self).__init__(args,input_dim, hidden_dim, gpu)
        # hidden_dim is the dim of the input_dim of rnn
        self.args=args
        self.att_weights = nn.Parameter(torch.Tensor(1, input_dim),requires_grad=True)
        nn.init.xavier_uniform(self.att_weights.data)
        self.hidden_dim = hidden_dim*2
        self.Wself = nn.Linear(input_dim, self.hidden_dim)

        # if self.transform_useLstm:
        #     self.birnn_layer = LayerBiLSTM(args=args,
        #                                    input_dim=self.input_dim,
        #                                    hidden_dim=rnn_hidden_dim,
        #                                    gpu=gpu)


    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len, _ = input_tensor.shape

        # apply attention layer
        att_weightss = self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([100, 850, 1])
        weights = torch.bmm(input_tensor,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )
        attentions = F.softmax(F.relu(weights.squeeze())) #(bs, max_seq_len)

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask_tensor #(bs, max_seq_len)
        _sums = masked.sum(-1,keepdim=True).expand_as(attentions)  # sums per row
        attentions = masked.div(_sums)

        # apply attention weights
        representation = torch.mul(input_tensor, attentions.unsqueeze(-1).expand_as(input_tensor))
        # print('representation.shape', representation.shape) # representation.shape torch.Size([100, 47, 850])

        representation = self.Wself(representation)


        return representation



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
