"""class implements character-level convolutional 1D layer"""
import string
import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.seq_indexers.seq_indexer_char import SeqIndexerBaseChar
class LayerCharLSTM(LayerBase):
    """LayerCharCNN implements character-level convolutional 1D layer."""
    def __init__(self, gpu, char_embeddings_dim, hidden_dim,word_len,unique_characters_list=None):
        super(LayerCharLSTM, self).__init__(gpu)
        self.char_embeddings_dim = char_embeddings_dim
        # self.char_cnn_filter_num = filter_nu
        # self.word_len = word_lenm
        # self.char_window_size = char_window_size
        # self.conv_feature_len = word_len - char_window_size + 1
        # self.output_dim = char_embeddings_dim * filter_num
        self.char_hidden_dim = hidden_dim
        self.word_len = word_len

        self.char_seq_indexer = SeqIndexerBaseChar(gpu=gpu)
        if unique_characters_list is None:
            unique_characters_list = list(string.printable)
        for c in unique_characters_list:
            self.char_seq_indexer.add_char(c)
        # Init character embedding
        self.embeddings = nn.Embedding(num_embeddings=self.char_seq_indexer.get_items_count(),
                                       embedding_dim=char_embeddings_dim,
                                       padding_idx=0)

        self.lstm_char = nn.LSTM(input_size=char_embeddings_dim,
                      hidden_size=hidden_dim,
                      num_layers=1,
                      batch_first=True,
                      bidirectional=True)
        self.Wh = nn.Linear(2 * self.char_hidden_dim, self.char_hidden_dim)
        self.output_dim = self.char_hidden_dim

    def is_cuda(self):
        return self.conv1d.weight.is_cuda


    def forward(self, char_embeddings_feature): # batch_num x max_seq_len x char_embeddings_dim x word_len
        # '''the forward input is word_sequences '''
        # batch_num = len(word_sequences)
        # max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        # char_sequences = [[[c for c in word] for word in word_seq] for word_seq in word_sequences]
        # input_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len, self.word_len, dtype=torch.long))
        # for n, curr_char_seq in enumerate(char_sequences):
        #     curr_seq_len = len(curr_char_seq)
        #     curr_char_seq_tensor = self.char_seq_indexer.get_char_tensor(curr_char_seq,
        #                                                                  self.word_len)  # curr_seq_len x word_len
        #     input_tensor[n, :curr_seq_len, :] = curr_char_seq_tensor
        # char_embeddings_feature = self.embeddings(input_tensor) #shape: batch_num x max_seq_len x word_len x char_embeddings_dim
        # char_hidden_dim = char_embeddings_feature.size(-1)
        # char_embeddings_feature_view = char_embeddings_feature.view(-1,self.word_len,char_hidden_dim)
        #
        # outputs, ht = self.lstm_char(char_embeddings_feature_view,batch_first=True)
        # ht = ht.permute(1,0,2).contiguous().view(batch_num,max_seq_len,-1)
        # ht = self.Wh(ht)
        #
        # return ht

        # ''' the input of forward is char_embeddings_feature '''
        # char_embeddings_feature # shape: batch_num x max_seq_len x char_embeddings_dim x word_len
        char_embeddings_feature = char_embeddings_feature.permute(0,1,3,2)
        batch_num, max_seq_len, word_len, char_embeddings_dim = char_embeddings_feature.shape
        char_embeddings_feature_view = char_embeddings_feature.view(-1,self.word_len, char_embeddings_dim)
        outputs, (ht, _) = self.lstm_char(char_embeddings_feature_view)
        ht = ht.permute(1,0,2).contiguous().view(batch_num, max_seq_len, -1)
        ht = self.Wh(ht)

        return ht



