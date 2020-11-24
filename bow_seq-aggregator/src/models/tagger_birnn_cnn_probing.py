"""BiLSTM/BiGRU + char-level CNN tagger model"""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.layers.layer_word_embeddings import LayerWordEmbeddings, LayerWordEmbeddings_Rand,LayerWordEmbeddings_ExtendVocab
from src.layers.layer_bivanilla import LayerBiVanilla
from src.layers.layer_bilstm import LayerBiLSTM
from src.layers.layer_bigru import LayerBiGRU
from src.layers.layer_char_embeddings import LayerCharEmbeddings
from src.layers.layer_elmo_embeddings import LayerElmoEmbeddings
from src.layers.layer_bert_embeddings import LayerBertEmbeddings
from src.layers.layer_flair_embeddings import LayerFlairEmbeddings
from src.layers.layer_char_cnn import LayerCharCNN
from src.layers.layer_char_lstm import LayerCharLSTM
import torch.nn.functional as F
import numpy as np
import pickle

class TaggerBiRNNCNN(TaggerBase):
    """TaggerBiRNNCNN is a model for sequences tagging that includes RNN and character-level conv-1D layer."""
    def __init__(self,args, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='GRU', gpu=-1,
                 freeze_char_embeddings = False, char_embeddings_dim=25, word_len=20, char_cnn_filter_num=30,
                 char_window_size=3):
        super(TaggerBiRNNCNN, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = True
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.freeze_char_embeddings = freeze_char_embeddings
        self.char_embeddings_dim = char_embeddings_dim
        self.word_len = word_len
        self.char_cnn_filter_num = char_cnn_filter_num
        self.char_window_size = char_window_size
        self.if_elmo = args.if_elmo
        self.if_bert = args.if_bert
        self.if_flair = args.if_flair
        if args.if_glove or args.if_wordEmbRand or args.if_twitter_emb:
            self.if_word = True
        else:
            self.if_word = False

        if args.if_char_cnn or args.if_char_lstm:
            self.if_char = True
        else:
            self.if_char = False
        # self.elmo_embeddings_dim = args.elmo_embeddings_dim
        # self.bert_embeddings_dim = args.bert_embeddings_dim
        self.bert_mode = 'mean'

        self.options_file = args.options_file
        self.weight_file = args.weight_file
        emb_models_dim = []

        if args.if_wordEmbRand:
            self.word_embeddings_layer = LayerWordEmbeddings_Rand(word_seq_indexer, gpu, self.freeze_embeddings)
            emb_models_dim.append(self.word_embeddings_layer.output_dim)
        if args.if_glove:
            self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, self.freeze_embeddings)
            emb_models_dim.append(self.word_embeddings_layer.output_dim)

        if args.if_char_lstm:
            self.char_embeddings_layer = LayerCharEmbeddings(gpu, char_embeddings_dim, freeze_char_embeddings,
                                                             word_len, word_seq_indexer.get_unique_characters_list())
            self.char_layer = LayerCharLSTM(gpu,char_embeddings_dim, self.char_lstm_hidden_dim, word_len)
            emb_models_dim.append(self.char_layer.output_dim)
        if args.if_char_cnn:
            self.char_embeddings_layer = LayerCharEmbeddings(gpu, char_embeddings_dim, freeze_char_embeddings,
                                                             word_len, word_seq_indexer.get_unique_characters_list())
            self.char_layer = LayerCharCNN(gpu, char_embeddings_dim, char_cnn_filter_num, char_window_size, word_len)
            emb_models_dim.append(self.char_layer.output_dim)
        if args.if_elmo:
            self.elmo_embeddings_layer = LayerElmoEmbeddings(args, gpu, self.elmo_embeddings_dim, self.options_file, self.weight_file,freeze_char_embeddings, word_len)
            emb_models_dim.append(self.elmo_embeddings_layer.output_dim)
        if args.if_bert:
            self.bert_embeddings_layer = LayerBertEmbeddings(gpu, self.bert_embeddings_dim, self.bert_mode)
            emb_models_dim.append(self.bert_embeddings_layer.output_dim)
            print('start bert embedding successful...')
        if args.if_flair:
            self.flair_embeddings_layer = LayerFlairEmbeddings(gpu)
            emb_models_dim.append(self.flair_embeddings_layer.output_dim)

        self.input_dim = sum(emb_models_dim)

        #
        # self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        self.char_embeddings_layer = LayerCharEmbeddings(gpu, char_embeddings_dim, freeze_char_embeddings,
                                                         word_len, word_seq_indexer.get_unique_characters_list())
        self.char_cnn_layer = LayerCharCNN(gpu, char_embeddings_dim, char_cnn_filter_num, char_window_size,
                                           word_len)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'GRU':
            self.birnn_layer = LayerBiGRU(args=args,
                                          input_dim=self.input_dim,
                                          hidden_dim=rnn_hidden_dim,
                                          gpu=gpu)
        elif rnn_type == 'LSTM':
            self.birnn_layer = LayerBiLSTM(args=args,
                                           input_dim=self.input_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        elif rnn_type == 'Vanilla':
            self.birnn_layer = LayerBiVanilla(args=args,
                                              input_dim=self.input_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num + 1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.nll_loss = nn.NLLLoss(ignore_index=0)  # "0" target values actually are zero-padded parts of sequences

    def forward(self, word_sequences):
        print('word_sequences',word_sequences)
        z = torch.tensor([0])
        if self.if_elmo == False and self.if_bert == False and self.if_char == False and self.if_word == True:
            z_word_embed = self.word_embeddings_layer(word_sequences)

            # replace with a random word...
            # print('z_word_embed[0][5], original', z_word_embed[0][2])
            # scale = np.sqrt(4.0 / 100)
            # random_word_emb = np.random.uniform(-scale, scale, [1, 100])
            # # random_word_emb = (np.random.random(size=100)-0.5)/100.0
            # print('random_word_emb', random_word_emb)
            # z_word_embed[0][4][:] = torch.Tensor(random_word_emb).unsqueeze(0)
            # print('z_word_embed[0][5], new', z_word_embed[0][4])

            # replace by a glove word ...
            # fread = open('emb/conll03_vocab_emb.txt', 'rb')
            # unique_words, emb_vecs = pickle.load(fread)
            # glove_word_emb =[]
            # if 'against'in unique_words :
            #     idx = unique_words.index('against')
            #     glove_word_emb = emb_vecs[idx]
            #     print('111')
            # if 'against' in unique_words:
            #     idx = unique_words.index('against')
            #     glove_word_emb = emb_vecs[idx]
            #     print('222')
            # print('z_word_embed[0][2], original', z_word_embed[1][14])
            # z_word_embed[1][14][:] = torch.Tensor(glove_word_emb).unsqueeze(0)
            # print('z_word_embed[0][2], new', z_word_embed[1][14])

            # replace with a glove matrix..
            # fread = open('emb/conll03_vocab_emb.txt', 'rb')
            # unique_words, emb_vecs = pickle.load(fread)

            # glove_words = []
            # glove_embs = []
            # for k, line in enumerate(open('../embed/glove/glove.6B.100d.txt', 'r')):
            #     values = line.split()
            #     glove_words.append(values[0])
            #     emb = [float(x) for x in values[1:]]
            #     glove_embs.append(emb)
            #
            #
            # for i,words in enumerate(word_sequences):
            #     for j,word in enumerate(words):
            #         if word in glove_words:
            #             glove_word_emb = glove_embs[glove_words.index(word)]
            #             # print('glove_word_emb',glove_word_emb)
            #         elif word.lower() in glove_words:
            #             glove_word_emb = glove_embs[glove_words.index(word.lower())]
            #             # print('glove_word_emb', glove_word_emb)
            #         else:
            #             scale = np.sqrt(3.0 / 100)
            #             glove_word_emb = np.random.uniform(-scale, scale, [1, 100])
            #         z_word_embed[i][j][:] = torch.Tensor(glove_word_emb).unsqueeze(0)



            # # random embeding matrix ...
            # scale = np.sqrt(3.0 / 100)
            # for i,words in enumerate(word_sequences):
            #     for j,word in enumerate(words):
            #         new_word_emb = np.random.uniform(-scale, scale, [1, 100])
            #         z_word_embed[i][j][:] = torch.Tensor(new_word_emb).unsqueeze(0)






            z = self.dropout(z_word_embed)
        mask = self.get_mask_from_word_sequences(word_sequences)  # batch_size x max_seq_len
        rnn_output_h = self.birnn_layer(z, mask)
        z_rnn_out = self.apply_mask(self.lin_layer(rnn_output_h), mask)
        pred = F.softmax(z_rnn_out, dim=-1)  # (bs, max_seq_len, 10_class)
        y = self.log_softmax_layer(z_rnn_out.permute(0, 2, 1))  # (batch,?_hidden, max_seq_len)

        return y,pred

        # mask = self.get_mask_from_word_sequences(word_sequences)
        # z_word_embed = self.word_embeddings_layer(word_sequences)
        # z_char_embed = self.char_embeddings_layer(word_sequences)
        # z_char_embed_d = self.dropout(z_char_embed)
        # z_char_cnn_d = self.dropout(self.char_cnn_layer(z_char_embed_d))
        # z = torch.cat((z_word_embed, z_char_cnn_d), dim=2)
        # rnn_output_h = self.birnn_layer(z, mask)
        # rnn_output_h_d = self.dropout(rnn_output_h) # shape: batch_size x max_seq_len x rnn_hidden_dim*2
        # z_rnn_out = self.apply_mask(self.lin_layer(rnn_output_h_d), mask)
        # y = self.log_softmax_layer(z_rnn_out.permute(0, 2, 1))
        # return y

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        outputs_tensor_train_batch_one_hot,pred = self.forward(word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        loss = self.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)
        return loss

    def get_prob(self,word_sequences_train_batch,tag_sequences_train_batch):
        outputs_tensor_train_batch_one_hot, pred = self.forward(word_sequences_train_batch)
        targets_indx_train_batch = self.tag_seq_indexer.items2idx(tag_sequences_train_batch)
        total_probs =[]
        for i in range(len(targets_indx_train_batch)):
            probs = []
            for j in range(len(targets_indx_train_batch[i])):
                probs.append(pred[i][j][targets_indx_train_batch[i][j]].item())
            total_probs.append(probs)
        return total_probs

