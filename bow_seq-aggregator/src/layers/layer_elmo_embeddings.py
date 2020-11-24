"""class implements character-level embeddings"""
import string
import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase
from src.seq_indexers.seq_indexer_char import SeqIndexerBaseChar
from allennlp.modules.elmo import Elmo, batch_to_ids
# from elmoformanylangs import Embedder

class LayerElmoEmbeddings(LayerBase):
    """LayerElmoEmbeddings implements pretrained elmo embeddings."""
    def __init__(self, args, gpu, options_file, weight_file, freeze_char_embeddings=False, word_len=20):
        super(LayerElmoEmbeddings, self).__init__(gpu)
        self.gpu = gpu
        self.elmo_embeddings_dim = 100
        self.freeze_char_embeddings = freeze_char_embeddings
        self.word_len = word_len # standard len to pad
        # Init character sequences indexer
        self.char_seq_indexer = SeqIndexerBaseChar(gpu=gpu)
        # if unique_characters_list is None:
        #     unique_characters_list = list(string.printable)
        # for c in unique_characters_list:
        #     self.char_seq_indexer.add_char(c)
        # Init character embedding
        # self.embeddings = nn.Embedding(num_embeddings=self.char_seq_indexer.get_items_count(),
        #                                embedding_dim=char_embeddings_dim,
        #                                padding_idx=0)
        # nn.init.uniform_(self.embeddings.weight, -0.5, 0.5) # Option: Ma, 2016
        # self.Welmo = nn.Linear(256,elmo_embeddings_dim)
        if args.if_elmo_large:
            self.output_dim = 1024
        else:
            self.output_dim = 256

        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)

        # self.elmo = Embedder('../Projects/ELMo_Chinese/zhs_model/')

    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, word_sequences):
        # elmo = Embedder('../Projects/ELMo_Chinese2/zhs_model/')
        # embeddings = elmo.sents2elmo(word_sequences)
        # # embeddings = self.elmo.sents2elmo(word_sequences)
        # print('embeddings[0]',embeddings[0])
        # print('embeddings[1]', embeddings[1])
        # print('embeddings[2]', embeddings[2])
        # print()
        # print('embeddings.shape',embeddings.shape)

        character_ids = batch_to_ids(word_sequences)
        embeddings = self.elmo(character_ids.cuda())['elmo_representations']
        embeddings = (embeddings[0]+embeddings[1])/2.0

        return embeddings
