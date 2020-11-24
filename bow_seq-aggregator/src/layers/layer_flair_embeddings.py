"""class implements character-level embeddings"""
import string
import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase
from src.seq_indexers.seq_indexer_char import SeqIndexerBaseChar

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from flair.embeddings import FlairEmbeddings,StackedEmbeddings,BertEmbeddings,WordEmbeddings
from flair.data import Sentence
import  torch.nn

class LayerFlairEmbeddings(LayerBase):
    """LayerBertEmbeddings implements character-level embeddings."""
    def __init__(self,gpu):
        super(LayerFlairEmbeddings, self).__init__(gpu)
        self.gpu = gpu
        # self.flair_embeddings_dim = flair_embeddings_dim
        # self.freeze_flair_embeddings = freeze_flair_embeddings

        self.output_dim = 4096

        self.flair_embedding_forward = FlairEmbeddings('/home/jlfu/flair_model/news-forward-0.4.1.pt')
        self.flair_embedding_backward = FlairEmbeddings('/home/jlfu/flair_model/news-backward-0.4.1.pt')
        self.stacked_embeddings = StackedEmbeddings([
            self.flair_embedding_forward,
            self.flair_embedding_backward
        ])
        # self.glove_embedding = WordEmbeddings('glove')
        # self.args= args
        # if self.args.use_flair_glove:
        #     self.stacked_embeddings = StackedEmbeddings([
        #         self.glove_embedding,
        #         self.flair_embedding_forward,
        #         self.flair_embedding_backward
        #     ])
        #     self.output_dim = 4096



    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, word_sequences):
        batch_size = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        flair_embedding = torch.zeros(batch_size, max_seq_len, self.output_dim)

        # create a sentence
        for i,word_sequence in enumerate(word_sequences):
            word_seq_str = ' '.join(word_sequence)
            sentence = Sentence(word_seq_str)
            # self.flair_embedding_forward.embed(sentence)
            self.stacked_embeddings.embed(sentence)
            for j,token in enumerate(sentence):
                # print('token.embedding',token.embedding)
                flair_embedding[i][j][:] = token.embedding
            # print('flair_embedding',flair_embedding)
            # break
        return flair_embedding


    # def forward(self, word_sequences):
    #     batch_size = len(word_sequences)
    #     max_seq_len = max([len(word_seq) for word_seq in word_sequences])
    #     flair_embedding = torch.zeros(batch_size, max_seq_len, 2048)
    #
    #     # init embedding
    #     # flair_embedding_forward = FlairEmbeddings('news-forward')
    #     # flair_embedding_forward = FlairEmbeddings('news-backward')
    #     flair_embedding_forward = FlairEmbeddings('/home/jlfu/flair_model/news-backward-0.4.1.pt')
    #     flair_embedding_backward = FlairEmbeddings('/home/jlfu/flair_model/news-backward-0.4.1.pt')
    #     stacked_embeddings = StackedEmbeddings([
    #         # WordEmbeddings('glove'),
    #         # FlairEmbeddings('news-forward'),
    #         # FlairEmbeddings('news-backward'),
    #         flair_embedding_forward,
    #         flair_embedding_backward
    #     ])
    #
    #     # create a sentence
    #     for i,word_sequence in enumerate(word_sequences):
    #         word_seq_str = ' '.join(word_sequence)
    #         sentence = Sentence(word_seq_str)
    #         # flair_embedding_forward.embed(sentence)
    #
    #         stacked_embeddings.embed(sentence)
    #         for j,token in enumerate(sentence):
    #             # print('token.embedding',token.embedding)
    #             flair_embedding[i][j][:] = token.embedding
    #         # print('flair_embedding',flair_embedding)
    #         # break
    #     return flair_embedding







# if __name__ == '__main__':
# 	flairEmb = LayerFlairEmbeddings(gpu=True, flair_embeddings_dim=100, freeze_flair_embeddings=True)
# 	sentence = [['I', 'love', 'Berlin', '.']]
# 	flairEmb(sentence)

        #
        #
        # bert = self.bert
        # bert.cuda()
        #
        # batch_size = len(word_sequences)
        # max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        # bert_embedding = torch.zeros(batch_size, max_seq_len, 768)
        #
        # for i, orig_tokens in enumerate(word_sequences):
        #     bert_tokens = []
        #     orig_to_tok_map = []
        #     bert_tokens.append('[CLS]')
        #     for orig_token in orig_tokens:
        #         orig_to_tok_map.append(len(bert_tokens))
        #         bert_tokens.extend(self.tokenizer.tokenize(orig_token))
        #     bert_tokens.append("[SEP]")
        #     indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        #
        #     segments_ids = [0 for m in range(len(bert_tokens))]
        #     segments_ids = torch.tensor([segments_ids],dtype=torch.long).cuda()
        #     indexed_tokens = torch.tensor([indexed_tokens],dtype=torch.long).cuda()
        #     input_mask = [1] * len(indexed_tokens)
        #     input_mask = torch.tensor([input_mask],dtype=torch.long).cuda()
        #
        #     encoded_layers, _ = bert(indexed_tokens, segments_ids,input_mask)
        #     orig_bert_hidden = encoded_layers[-1].squeeze(0)
        #
        #     # print('orig_to_tok_map', orig_to_tok_map)  # ([1, 3, 6, 7, 8])
        #     if mode == 'first':
        #         bert_hidden = orig_bert_hidden[orig_to_tok_map]  # torch.Size([5, 7, 768])
        #         bert_embedding[i][:len(bert_hidden)] = bert_hidden
        #
        #     # subword embedding mean ...
        #     if mode == 'mean':
        #         bert_hidden = []
        #         for j, idx in enumerate(orig_to_tok_map):
        #             if j < len(orig_to_tok_map) - 1:
        #                 bert_hid = torch.mean(orig_bert_hidden[orig_to_tok_map[j]:orig_to_tok_map[j + 1]], dim=0,
        #                                       keepdim=False)
        #                 bert_hidden.append(bert_hid)
        #             else:
        #                 bert_hid = torch.mean(orig_bert_hidden[orig_to_tok_map[j]:-1], dim=0, keepdim=False)
        #                 bert_hidden.append(bert_hid)
        #             # print('bert_hidden[j][:10]', bert_hidden[j][:10])
        #         bert_embedding[i][:len(bert_hidden)] = torch.stack(bert_hidden)
            # print('orig_bert_hidden', orig_bert_hidden)
        # return bert_embedding

