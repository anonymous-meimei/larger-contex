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

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.optimization import warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer



class LayerBertEmbeddings(BertForTokenClassification):
    """LayerBertEmbeddings implements character-level embeddings."""
    # def __init__(self, gpu, freeze_bert_embeddings=True):
    #     super(LayerBertEmbeddings, self).__init__(gpu)
    #     self.gpu = gpu
    #     # self.bert_embeddings_dim = bert_embeddings_dim
    #     self.freeze_char_embeddings = freeze_bert_embeddings
        # self.bert = BertModel.from_pretrained("/home/jlfu/model/cased_L-12_H-768_A-12/bert-base-cased.tar.gz")
        # self.tokenizer = BertTokenizer.from_pretrained('/home/jlfu/model/cased_L-12_H-768_A-12/vocab.txt')

        # self.Wbert = nn.Linear(768, bert_embeddings_dim)
        # self.output_dim = 768
        # self.bert_embedding = BertEmbeddings("bert-base-cased")  # bert-base-cased,  bert-base-multilingual-cased
        # self.stack_embedding = StackedEmbeddings(embeddings = [self.bert_embedding])
    def forward(self, word_sequences, input_bert):
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])


        input_ids, input_masks, segment_ids, label_ids, valid_ids, label_masks = input_bert

        #(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                    # attention_mask_label=None)
            # input_ids, segment_ids, input_masks, label_ids, valid_ids, l_mask
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_masks = torch.tensor(input_masks, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        valid_ids = torch.tensor(valid_ids, dtype=torch.long)
        # label_masks = torch.tensor(label_masks, dtype=torch.long)
        sequence_output, _ = self.bert(input_ids.cuda(),segment_ids.cuda(),input_masks.cuda(),output_all_encoded_layers=False)
        print('load bert success ...')
        batch_size,max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_seq_len,feat_dim,dtype=torch.float32,device='cuda')
        for i in range(batch_size):
            jj=-1
            for j in range(max_len):
                if valid_ids[i][j].item()==1:
                    jj+=1
                    valid_output[i][jj] = sequence_output[i][j]
                    if jj ==max_seq_len-1:
                        break
        # sequence_output = self.dropout(valid_output)
        # logits = self.classifier(sequence_output)

        return valid_output


