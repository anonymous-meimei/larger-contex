"""class implements word embeddings"""
import torch.nn as nn
from src.layers.layer_base import LayerBase
from src.factories.factory_datasets_bank import DatasetsBankFactory
import numpy as np
import torch

class LayerWordEmbeddings(LayerBase):
    """LayerWordEmbeddings implements word embeddings."""
    def __init__(self,args, word_seq_indexer, gpu, freeze_word_embeddings=False, pad_idx=0):
        super(LayerWordEmbeddings, self).__init__(gpu)
        embeddings_tensor = word_seq_indexer.get_loaded_embeddings_tensor()
        self.embeddings = nn.Embedding.from_pretrained(embeddings=embeddings_tensor, freeze=freeze_word_embeddings)
        self.embeddings.padding_idx = pad_idx
        self.word_seq_indexer = word_seq_indexer
        self.freeze_embeddings = freeze_word_embeddings
        self.embeddings_num = embeddings_tensor.shape[0]
        self.embeddings_dim = embeddings_tensor.shape[1]
        # self.output_dim = self.embeddings_dim
        if args.if_no_bigram:
            self.output_dim = self.embeddings_dim * 5
        else:
            self.output_dim = self.embeddings_dim*9
    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, word_sequences):
        new_word_sequences = []
        for word_seq in word_sequences:
            new_word_seq = []
            for win_seq in word_seq:
                new_word_seq += win_seq
            new_word_sequences.append(new_word_seq)

        input_tensor = self.tensor_ensure_gpu(self.word_seq_indexer.items2tensor(new_word_sequences)) # shape: batch_size x max_seq_len
        word_embeddings_feature = self.embeddings(input_tensor) # shape: batch_size x max_seq_len x output_dim
        batch_size = word_embeddings_feature.size(0)
        dim = word_embeddings_feature.size(-1)

        word_embeddings_feature = word_embeddings_feature.reshape(batch_size,-1,self.output_dim)
        return word_embeddings_feature

class LayerWordEmbeddings_Rand(LayerBase):
    """LayerWordEmbeddings_Rand, randomly initialized word embeddings."""
    def __init__(self, word_seq_indexer, gpu, freeze_word_embeddings=False, pad_idx=0):
        super(LayerWordEmbeddings_Rand, self).__init__(gpu)
        self.word_seq_indexer = word_seq_indexer
        self.embedding_dim=100
        rand_embeddings_tensor = word_seq_indexer.get_random_embedding(embedding_dim=self.embedding_dim)
        # print('rand_embeddings_tensor',rand_embeddings_tensor)
        self.embeddings = nn.Embedding.from_pretrained(embeddings= torch.FloatTensor(rand_embeddings_tensor), freeze=freeze_word_embeddings)
        self.embeddings.padding_idx = pad_idx
        self.word_seq_indexer = word_seq_indexer
        self.freeze_embeddings = freeze_word_embeddings
        self.embeddings_num = rand_embeddings_tensor.shape[0]
        self.embeddings_dim = rand_embeddings_tensor.shape[1]
        self.output_dim = self.embeddings_dim

    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, word_sequences):
        # word_sequences = word_sequences.reshape(word_sequences.size(0),-1)
        new_word_sequences = []
        for word_seq in word_sequences:
            new_word_seq = []
            for win_seq in word_seq:
                new_word_seq+=win_seq
            new_word_sequences.append(new_word_seq)



        input_tensor = self.tensor_ensure_gpu(self.word_seq_indexer.items2tensor(new_word_sequences)) # shape: batch_size x max_seq_len
        # print('input_tensor',input_tensor)
        word_embeddings_feature = self.embeddings(input_tensor) # shape: batch_size x max_seq_len x output_dim
        return word_embeddings_feature

class LayerWordEmbeddings_ExtendVocab(LayerBase):
    """LayerWordEmbeddings implements word embeddings."""
    def __init__(self, word_seq_indexer, gpu, freeze_word_embeddings=False, pad_idx=0):
        super(LayerWordEmbeddings_ExtendVocab, self).__init__(gpu)

        embeddings_tensor = word_seq_indexer.get_loaded_embeddings_tensor()
        embedding_dim = len(embeddings_tensor[0])
        pretraind_embeddings_weights = word_seq_indexer.get_pretrained_embeddingsWeights()[1]
        embeddings_weights = torch.Tensor(len(embeddings_tensor), embedding_dim)
        embeddings_weights[:len(pretraind_embeddings_weights),:] =pretraind_embeddings_weights

        self.embeddings.padding_idx = pad_idx
        randIni_embeddings_weights = torch.Tensor(len(embeddings_tensor)-len(pretraind_embeddings_weights), embedding_dim)
        randIni_embeddings_weights.data.normal_(0, 1)
        if self.embeddings.padding_idx is not None:
            randIni_embeddings_weights.data[self.embeddings.padding_idx].fill_(0)
        embeddings_weights[len(pretraind_embeddings_weights):, :] =randIni_embeddings_weights

        self.embeddings = nn.Embedding.from_pretrained(embeddings=embeddings_tensor, freeze=freeze_word_embeddings)
        self.embeddings.weight.data.copy_(torch.from_numpy(embeddings_weights))

        self.word_seq_indexer = word_seq_indexer
        self.freeze_embeddings = freeze_word_embeddings
        self.embeddings_num = embeddings_tensor.shape[0]
        self.embeddings_dim = embeddings_tensor.shape[1]
        self.output_dim = self.embeddings_dim

    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, word_sequences):
        input_tensor = self.tensor_ensure_gpu(self.word_seq_indexer.items2tensor(word_sequences)) # shape: batch_size x max_seq_len
        word_embeddings_feature = self.embeddings(input_tensor) # shape: batch_size x max_seq_len x output_dim
        return word_embeddings_feature
