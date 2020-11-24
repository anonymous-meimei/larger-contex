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
from flair.embeddings import FlairEmbeddings,StackedEmbeddings,BertEmbeddings
from flair.data import Sentence

class LayerBertEmbeddings(LayerBase):
    """LayerBertEmbeddings implements character-level embeddings."""
    def __init__(self, gpu, bert_embeddings_dim, freeze_bert_embeddings=True):
        super(LayerBertEmbeddings, self).__init__(gpu)
        self.gpu = gpu
        self.bert_embeddings_dim = bert_embeddings_dim
        self.freeze_char_embeddings = freeze_bert_embeddings

        # self.bert = BertModel.from_pretrained("/home/jlfu/saved_pytorch_bert/en_base_uncased/model")
        # # for p in self.bert.parameters():
        # #     p.requires_grad = True
        # self.tokenizer = BertTokenizer.from_pretrained('/home/jlfu/saved_pytorch_bert/en_base_uncased/vocab.txt')

        # self.bert = BertModel.from_pretrained("/home/jlfu/model/cased_L-12_H-768_A-12/bert_model.ckpt.gz")
        # self.bert = BertModel.from_pretrained('bert-base-cased')

        self.bert = BertModel.from_pretrained("/home/jlfu/model/cased_L-12_H-768_A-12/bert-base-cased.tar.gz")
        self.tokenizer = BertTokenizer.from_pretrained('/home/jlfu/model/cased_L-12_H-768_A-12/vocab.txt')

        self.Wbert = nn.Linear(768, bert_embeddings_dim)
        self.output_dim = 768
        
        # self.output_dim =  3072

        self.bert_embedding = BertEmbeddings("bert-base-cased")  # bert-base-cased,  bert-base-multilingual-cased
        # self.stack_embedding = StackedEmbeddings(embeddings = [self.bert_embedding])
    def is_cuda(self):
        return self.embeddings.weight.is_cuda
    # def freeze_bert_encoder(self):
    #     for p in self.bert.parameters():
    #         p.requires_grad = False
    #
    # def unfreeze_bert_encoder(self):
    #     for p in self.bert.parameters():
    #         p.requires_grad = True

    # def forward(self,word_sequences,mode='first'):
    #    batch_size = len(word_sequences)
    #    max_seq_len = max([len(word_seq) for word_seq in word_sequences])
    #    bert_emb = torch.zeros(batch_size, max_seq_len, 3072)
    #
    #    # create a sentence
    #    for i,word_sequence in enumerate(word_sequences):
    #        word_seq_str = ' '.join(word_sequence)
    #        sentence = Sentence(word_seq_str)
    #        # self.flair_embedding_forward.embed(sentence)
    #        self.bert_embedding.embed(sentence)
    #        for j,token in enumerate(sentence):
    #            # print('token.embedding',token.embedding)
    #            # print('token.embedding.shape', token.embedding.shape)
    #            bert_emb[i][j][:] = token.embedding
    #
    #        # print('flair_embedding',flair_embedding)
    #        # break
    #    return bert_emb









    def forward(self, word_sequences, mode='first'):
        bert = self.bert
        bert.cuda()
        subtoken_word_sequences = []

        batch_size = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        bert_embedding = torch.zeros(batch_size, max_seq_len, 768)
        orig_to_tok_maps = []
        for i, orig_tokens in enumerate(word_sequences):
            bert_tokens = []
            orig_to_tok_map = []
            bert_tokens.append('[CLS]')
            for orig_token in orig_tokens:
                orig_to_tok_map.append(len(bert_tokens))
                bert_tokens.extend(self.tokenizer.tokenize(orig_token))
            bert_tokens.append("[SEP]")
            subtoken_word_sequences.append(bert_tokens)
            orig_to_tok_maps.append(orig_to_tok_map)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)

            segments_ids = [0 for m in range(len(bert_tokens))]
            segments_ids = torch.tensor([segments_ids],dtype=torch.long).cuda()
            indexed_tokens = torch.tensor([indexed_tokens],dtype=torch.long).cuda()
            input_mask = [1] * len(indexed_tokens)
            input_mask = torch.tensor([input_mask],dtype=torch.long).cuda()

            encoded_layers, _ = bert(indexed_tokens, segments_ids,input_mask)

            # utilizing only the last layer...
            orig_bert_hidden = encoded_layers[-1].squeeze(0)

            # # utilizing the last four layers ...
            # bert_lastFourLayer_sum = torch.add(encoded_layers[-1],encoded_layers[-2])
            # bert_lastFourLayer_sum = torch.add(bert_lastFourLayer_sum, encoded_layers[-3])
            # bert_lastFourLayer_sum = torch.add(bert_lastFourLayer_sum, encoded_layers[-4])
            # orig_bert_hidden = bert_lastFourLayer_sum.squeeze(0)

            if mode == 'first':
                bert_hidden = orig_bert_hidden[orig_to_tok_map]  # torch.Size([5, 7, 768])
                bert_embedding[i][:len(bert_hidden)] = bert_hidden
            # # subword embedding mean ...
            if mode == 'mean':
                bert_hidden = []
                for j, idx in enumerate(orig_to_tok_map):
                    if j < len(orig_to_tok_map) - 1:
                        bert_hid = torch.mean(orig_bert_hidden[orig_to_tok_map[j]:orig_to_tok_map[j + 1]], dim=0,
                                              keepdim=False)
                        bert_hidden.append(bert_hid)
                    else:
                        bert_hid = torch.mean(orig_bert_hidden[orig_to_tok_map[j]:-1], dim=0, keepdim=False)
                        bert_hidden.append(bert_hid)
                    # print('bert_hidden[j][:10]', bert_hidden[j][:10])
                bert_embedding[i][:len(bert_hidden)] = torch.stack(bert_hidden)

            # put all the tokens into the lstm layer...
            # bert_embedding[i][:len(orig_bert_hidden)] = orig_bert_hidden

        return bert_embedding


            # if mode == 'first':
            #     bert_hidden = orig_bert_hidden[orig_to_tok_map]  # torch.Size([5, 7, 768])
            #     bert_embedding[i][:len(bert_hidden)] = bert_hidden
            #
            # # subword embedding mean ...
            # if mode == 'mean':
            #     bert_hidden = []
            #     for j, idx in enumerate(orig_to_tok_map):
            #         if j < len(orig_to_tok_map) - 1:
            #             bert_hid = torch.mean(orig_bert_hidden[orig_to_tok_map[j]:orig_to_tok_map[j + 1]], dim=0,
            #                                   keepdim=False)
            #             bert_hidden.append(bert_hid)
            #         else:
            #             bert_hid = torch.mean(orig_bert_hidden[orig_to_tok_map[j]:-1], dim=0, keepdim=False)
            #             bert_hidden.append(bert_hid)
            #         # print('bert_hidden[j][:10]', bert_hidden[j][:10])
            #     bert_embedding[i][:len(bert_hidden)] = torch.stack(bert_hidden)
            #
        # return bert_embedding,orig_to_tok_maps,subtoken_word_sequences

