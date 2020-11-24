"""converts list of lists of words as strings to list of lists of integer indices and back"""
import string
import re
import os
import pickle
from src.seq_indexers.seq_indexer_base_embeddings import SeqIndexerBaseEmbeddings
import numpy as np
from gensim.models import word2vec
# from word2vecReader import word2Vec
# from numpy import random
import torch
import codecs


class SeqIndexerWord(SeqIndexerBaseEmbeddings):
    """SeqIndexerWord converts list of lists of words as strings to list of lists of integer indices and back."""
    def __init__(self,args, gpu=-1, check_for_lowercase=True, embeddings_dim=0, verbose=True,unique_words_list=None):
        SeqIndexerBaseEmbeddings.__init__(self, gpu=gpu, check_for_lowercase=check_for_lowercase, zero_digits=True,
                                          pad='<pad>', unk='<unk>', load_embeddings=True, embeddings_dim=embeddings_dim,
                                          verbose=verbose)
        self.original_words_num = 0
        self.lowercase_words_num = 0
        self.zero_digits_replaced_num = 0
        self.zero_digits_replaced_lowercase_num = 0
        self.capitalize_word_num = 0
        self.uppercase_word_num = 0
        self.unique_words_list = unique_words_list
        self.args = args

    # def get_chinese_bigram_embeddings(self, vocab_emb_fn, emb_fn, unique_words_list, window_size):





    def load_chinese_character_embeddings_likeW2v(self, vocab_emb_fn, emb_fn, unique_words_list):
        if os.path.exists(vocab_emb_fn):
            print('load pre-trained word embedding...')
            fread = open(vocab_emb_fn, 'rb')
            unique_words, emb_vecs = pickle.load(fread)
            for unique_word, emb_vec in zip(unique_words, emb_vecs):
                self.add_word_emb_vec(unique_word, emb_vec)
        else:
            print('len(unique_words_list)', len(unique_words_list))
            # emb_word_dict2unique_word_list = dict()
            # '../Projects/twitter_w2v/40Wtweet_200dim.model'
            out_of_vocabulary_words_list = list()
            model = gensim.models.Word2Vec.load(emb_fn)
            # print('vocab:',model.vocab.keys() )

            # Add pretrained embeddings for unique_words
            unique_words = []
            emb_vecs = []
            for unique_word in unique_words_list:
                # print('unique_word',unique_word)
                emb_vec =[]
                try:
                    emb_vec = model[unique_word]
                except KeyError:
                    try:
                        emb_vec = model[unique_word.encode('utf8')]
                        self.original_words_num +=1

                    except KeyError:
                        out_of_vocabulary_words_list.append(unique_word)
                        self.lowercase_words_num +=1
                        # print('not find...')
                if len(emb_vec) >2:
                    print('add embedding...')
                    self.add_word_emb_vec(unique_word, emb_vec)
                    unique_words.append(unique_word)
                    emb_vecs.append(emb_vec)
            print('len(unique_words)', len(unique_words))
            fwrite = open(vocab_emb_fn, 'wb')
            pickle.dump([unique_words, emb_vecs], fwrite)
            fwrite.close()
            del unique_words
            del emb_vecs

            if self.verbose:
                print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
                print('    First 50 OOV words:')
                for i, oov_word in enumerate(out_of_vocabulary_words_list):
                    print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                    if i > 49:
                        break
                print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
                print(' -- original_words_num = %d' % self.original_words_num)
                print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
                print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
                print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)


    def load_chinese_character_embeddings_randomInitialize(self, vocab_emb_fn, emb_fn, emb_delimiter, emb_load_all,
                                                              unique_words_list):
        if os.path.exists(vocab_emb_fn):
            print('load pre-trained word embedding...')
            fread = open(vocab_emb_fn, 'rb')
            unique_words, emb_vecs = pickle.load(fread)
            for unique_word, emb_vec in zip(unique_words, emb_vecs):
                self.add_word_emb_vec(unique_word, emb_vec)
        else:
            # char_vecs = []
            # emb_chars = []
            # char2emb = {}
            # with codecs.open(emb_fn, 'r', 'utf-8') as f:
            #     lines = f.readlines()
            # for line in lines:
            #     # print('line',line)
            #     line =line.strip().split()
            #     vector = np.asarray(list(map(float,line[1:])))
            #     char_vecs.append(vector)
            #     emb_char = line[0]
            #     emb_chars.append(emb_char)
            #     char2emb[emb_char]=vector

            embedding_dim = 100
            char2emb = {}
            scale = np.sqrt(3.0 / embedding_dim)
            for unique_word in unique_words_list:
                if len(unique_word) == 1:
                    char2emb[unique_word] = np.random.uniform(-scale, scale, [embedding_dim])


            unique_words = []
            emb_vecs =[]
            out_of_vocabulary_words_list = list()
            for unique_word in unique_words_list:
                if len(unique_word)==1:
                    if unique_word in char2emb:
                        emb_vec = char2emb[unique_word]
                        self.add_word_emb_vec(unique_word, emb_vec)
                        unique_words.append(unique_word)
                        emb_vecs.append(emb_vec)
                        self.original_words_num+=1
                    else:
                        out_of_vocabulary_words_list.append(unique_word)

                elif len(unique_word)==2:
                    w1 = unique_word[0]
                    w2 = unique_word[1]
                    emb_w1 = []
                    emb_w2 = []
                    if w1 in char2emb:
                        emb_w1 = char2emb[w1]
                    else:
                        emb_w1 = char2emb['<OOV>']
                    if w2 in char2emb:
                        emb_w2 = char2emb[w2]
                    else:
                        emb_w2 = char2emb['<OOV>']
                    emb_vec = (emb_w1+emb_w2) / 2.0
                    self.add_word_emb_vec(unique_word, emb_vec)
                    unique_words.append(unique_word)
                    emb_vecs.append(emb_vec)

            fwrite = open(vocab_emb_fn, 'wb')
            pickle.dump([unique_words, emb_vecs], fwrite)
            fwrite.close()
            del unique_words
            del emb_vecs

            if self.verbose:
                print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
                print('    First 50 OOV words:')
                for i, oov_word in enumerate(out_of_vocabulary_words_list):
                    print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                    if i > 49:
                        break
                print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
                print(' -- original_words_num = %d' % self.original_words_num)



    def load_chinese_preTrainedCharBigram_embeddings_likeGlove(self, vocab_emb_fn, emb_fn,bigram_emb_fn, emb_delimiter, emb_load_all,
                                                              unique_words_list):
        if os.path.exists(vocab_emb_fn):
            print('load pre-trained word embedding...')
            fread = open(vocab_emb_fn, 'rb')
            unique_words, emb_vecs = pickle.load(fread)
            for unique_word, emb_vec in zip(unique_words, emb_vecs):
                self.add_word_emb_vec(unique_word, emb_vec)
        else:
            char_vecs = []
            emb_chars = []
            char2emb = {}

            bichar_vecs = []
            emb_bichars = []
            bichar2emb ={}
            with codecs.open(emb_fn, 'r', 'utf-8') as f:
                lines = f.readlines()
            for line in lines:
                # print('line',line)
                line =line.strip().split()
                vector = np.asarray(list(map(float,line[1:])))
                char_vecs.append(vector)
                emb_char = line[0]
                emb_chars.append(emb_char)
                char2emb[emb_char]=vector
            print('char vocabuary length is: ', len(char2emb))

            with codecs.open(bigram_emb_fn, 'r', 'utf-8') as f:
                lines = f.readlines()
            for line in lines:
                # print('line',line)
                line =line.strip().split()
                vector = np.asarray(list(map(float,line[1:])))
                bichar_vecs.append(vector)
                emb_bichar = line[0]
                emb_bichars.append(emb_bichar)
                bichar2emb[emb_bichar]=vector
            print('bichar vocabuary length is: ', len(bichar2emb))
            print('中国 embedding: ', bichar2emb['中国'])


            unique_words = []
            emb_vecs =[]
            out_of_vocabulary_words_list = list()
            for unique_word in unique_words_list:
                if len(unique_word)==1:
                    if unique_word in char2emb:
                        emb_vec = char2emb[unique_word]
                        self.add_word_emb_vec(unique_word, emb_vec)
                        unique_words.append(unique_word)
                        emb_vecs.append(emb_vec)
                        self.original_words_num+=1
                    else:
                        out_of_vocabulary_words_list.append(unique_word)

                elif len(unique_word)==2:
                    if unique_word in bichar2emb:
                        emb_vec = bichar2emb[unique_word]
                        self.add_word_emb_vec(unique_word, emb_vec)
                        unique_words.append(unique_word)
                        emb_vecs.append(emb_vec)
                        self.original_words_num+=1
                    else:
                        out_of_vocabulary_words_list.append(unique_word)


                    # w1 = unique_word[0]
                    # w2 = unique_word[1]
                    # emb_w1 = []
                    # emb_w2 = []
                    # if w1 in char2emb:
                    #     emb_w1 = char2emb[w1]
                    # else:
                    #     emb_w1 = char2emb['<OOV>']
                    # if w2 in char2emb:
                    #     emb_w2 = char2emb[w2]
                    # else:
                    #     emb_w2 = char2emb['<OOV>']
                    # emb_vec = (emb_w1+emb_w2) / 2.0
                    # self.add_word_emb_vec(unique_word, emb_vec)
                    # unique_words.append(unique_word)
                    # emb_vecs.append(emb_vec)

            fwrite = open(vocab_emb_fn, 'wb')
            pickle.dump([unique_words, emb_vecs], fwrite)
            fwrite.close()
            del unique_words
            del emb_vecs

            if self.verbose:
                print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
                print('    First 50 OOV words:')
                for i, oov_word in enumerate(out_of_vocabulary_words_list):
                    print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                    if i > 49:
                        break
                print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
                print(' -- original_words_num = %d' % self.original_words_num)


    def load_chinese_character_embeddings_likeGlove(self, vocab_emb_fn, emb_fn, emb_delimiter, emb_load_all,
                                                              unique_words_list=None):
        if os.path.exists(vocab_emb_fn):
            print('load pre-trained word embedding...')
            fread = open(vocab_emb_fn, 'rb')
            unique_words, emb_vecs = pickle.load(fread)
            for unique_word, emb_vec in zip(unique_words, emb_vecs):
                self.add_word_emb_vec(unique_word, emb_vec)
        else:
            char_vecs = []
            emb_chars = []
            char2emb = {}
            with codecs.open(emb_fn, 'r', 'utf-8') as f:
                lines = f.readlines()
            for line in lines:
                # print('line',line)
                line =line.strip().split()
                vector = np.asarray(list(map(float,line[1:])))
                char_vecs.append(vector)
                emb_char = line[0]
                emb_chars.append(emb_char)
                char2emb[emb_char]=vector

            unique_words = []
            emb_vecs =[]
            out_of_vocabulary_words_list = list()
            for unique_word in unique_words_list:
                if len(unique_word)==1:
                    if unique_word in char2emb:
                        emb_vec = char2emb[unique_word]
                        self.add_word_emb_vec(unique_word, emb_vec)
                        unique_words.append(unique_word)
                        emb_vecs.append(emb_vec)
                        self.original_words_num+=1
                    else:
                        out_of_vocabulary_words_list.append(unique_word)

                elif len(unique_word)==2:
                    w1 = unique_word[0]
                    w2 = unique_word[1]
                    emb_w1 = []
                    emb_w2 = []
                    if w1 in char2emb:
                        emb_w1 = char2emb[w1]
                    else:
                        emb_w1 = char2emb['<OOV>']
                    if w2 in char2emb:
                        emb_w2 = char2emb[w2]
                    else:
                        emb_w2 = char2emb['<OOV>']
                    emb_vec = (emb_w1+emb_w2) / 2.0
                    self.add_word_emb_vec(unique_word, emb_vec)
                    unique_words.append(unique_word)
                    emb_vecs.append(emb_vec)

            fwrite = open(vocab_emb_fn, 'wb')
            pickle.dump([unique_words, emb_vecs], fwrite)
            fwrite.close()
            del unique_words
            del emb_vecs

            if self.verbose:
                print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
                print('    First 50 OOV words:')
                for i, oov_word in enumerate(out_of_vocabulary_words_list):
                    print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                    if i > 49:
                        break
                print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
                print(' -- original_words_num = %d' % self.original_words_num)
                # print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
                # print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
                # print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)







            # embeddings_words_list = [emb_word for emb_word, _ in
            #                          SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn,
            #                                                                             emb_delimiter,
            #                                                                             verbose=True)]
            # # Create reverse mapping word from the embeddings file -> list of unique words from the dataset
            # # print('embeddings_words_list[:5]',embeddings_words_list[:5])
            # emb_word_dict2unique_word_list = dict()
            # out_of_vocabulary_words_list = list()
            # for unique_word in unique_words_list:
            #     emb_word = self.get_embeddings_word(unique_word, embeddings_words_list)
            #     if emb_word is None and len(emb_word)==1:
            #         out_of_vocabulary_words_list.append(unique_word)
            #     elif len(emb_word) ==2:
            #
            #
            #     else:
            #         if emb_word not in emb_word_dict2unique_word_list:
            #             emb_word_dict2unique_word_list[emb_word] = [unique_word]
            #         else:
            #             emb_word_dict2unique_word_list[emb_word].append(unique_word)
            # # Add pretrained embeddings for ,
            # unique_words = []
            # emb_vecs = []
            # for emb_word, emb_vec in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn, emb_delimiter,
            #                                                                             verbose=True):
            #     if emb_word in emb_word_dict2unique_word_list:
            #         for unique_word in emb_word_dict2unique_word_list[emb_word]:
            #             self.add_word_emb_vec(unique_word, emb_vec)
            #             unique_words.append(unique_word)
            #             emb_vecs.append(emb_vec)
            # fwrite = open(vocab_emb_fn, 'wb')
            # pickle.dump([unique_words, emb_vecs], fwrite)
            # fwrite.close()
            # del unique_words
            # del emb_vecs
            #
            # if self.verbose:
            #     print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
            #     print('    First 50 OOV words:')
            #     for i, oov_word in enumerate(out_of_vocabulary_words_list):
            #         print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
            #         if i > 49:
            #             break
            #     print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
            #     print(' -- original_words_num = %d' % self.original_words_num)
            #     print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
            #     print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
            #     print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)


    def load_twitter(self, vocab_emb_fn, emb_fn, unique_words_list):
        if os.path.exists(vocab_emb_fn):
            print('load pre-trained word embedding...')
            fread = open(vocab_emb_fn, 'rb')
            unique_words, emb_vecs = pickle.load(fread)
            for unique_word, emb_vec in zip(unique_words, emb_vecs):
                self.add_word_emb_vec(unique_word, emb_vec)
        else:
            print('len(unique_words_list)', len(unique_words_list))
            # emb_word_dict2unique_word_list = dict()
            # '../Projects/twitter_w2v/40Wtweet_200dim.model'
            out_of_vocabulary_words_list = list()
            model = word2vec.Word2Vec.load(emb_fn)

            # Add pretrained embeddings for unique_words
            unique_words = []
            emb_vecs = []
            for unique_word in unique_words_list:
                emb_vec =[]
                try:
                    emb_vec = model[unique_word.lower()]
                except KeyError:
                    try:
                        emb_vec = model[unique_word.lower().encode('utf8')]
                        self.original_words_num +=1
                    except KeyError:
                        out_of_vocabulary_words_list.append(unique_word)
                        self.lowercase_words_num +=1
                if len(emb_vec) >2:
                    self.add_word_emb_vec(unique_word, emb_vec)
                    unique_words.append(unique_word)
                    emb_vecs.append(emb_vec)
            print('len(unique_words)', len(unique_words))
            fwrite = open(vocab_emb_fn, 'wb')
            pickle.dump([unique_words, emb_vecs], fwrite)
            fwrite.close()
            del unique_words
            del emb_vecs

            if self.verbose:
                print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
                print('    First 50 OOV words:')
                for i, oov_word in enumerate(out_of_vocabulary_words_list):
                    print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                    if i > 49:
                        break
                print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
                print(' -- original_words_num = %d' % self.original_words_num)
                print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
                print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
                print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)


    def load_items_from_embeddings_file_and_unique_words_list(self, vocab_emb_fn, emb_fn, emb_delimiter, emb_load_all,
                                                              unique_words_list):
        # Get the full list of available case-sensitive words from text file with pretrained embeddings
        if os.path.exists(vocab_emb_fn):
            print('load pre-trained word embedding...')
            fread = open(vocab_emb_fn, 'rb')
            unique_words, emb_vecs = pickle.load(fread)
            for unique_word, emb_vec in zip(unique_words, emb_vecs):
                self.add_word_emb_vec(unique_word, emb_vec)

        else:
            embeddings_words_list = [emb_word for emb_word, _ in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn,
                                                                                                              emb_delimiter,
                                                                                                              verbose=True)]
            # Create reverse mapping word from the embeddings file -> list of unique words from the dataset
            emb_word_dict2unique_word_list = dict()
            out_of_vocabulary_words_list = list()
            for unique_word in unique_words_list:
                emb_word = self.get_embeddings_word(unique_word, embeddings_words_list)
                if emb_word is None:
                    out_of_vocabulary_words_list.append(unique_word)
                else:
                    if emb_word not in emb_word_dict2unique_word_list:
                        emb_word_dict2unique_word_list[emb_word] = [unique_word]
                    else:
                        emb_word_dict2unique_word_list[emb_word].append(unique_word)
            # Add pretrained embeddings for ,
            unique_words =[]
            emb_vecs = []
            for emb_word, emb_vec in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn, emb_delimiter,verbose=True):
                if emb_word in emb_word_dict2unique_word_list:
                    for unique_word in emb_word_dict2unique_word_list[emb_word]:
                        self.add_word_emb_vec(unique_word, emb_vec)
                        unique_words.append(unique_word)
                        emb_vecs.append(emb_vec)
            fwrite = open(vocab_emb_fn, 'wb')
            pickle.dump([unique_words,emb_vecs], fwrite)
            fwrite.close()
            del unique_words
            del emb_vecs


            if self.verbose:
                print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
                print('    First 50 OOV words:')
                for i, oov_word in enumerate(out_of_vocabulary_words_list):
                    print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                    if i > 49:
                        break
                print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
                print(' -- original_words_num = %d' % self.original_words_num)
                print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
                print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
                print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)

        # # Load all embeddings
        # if emb_load_all:
        #     loaded_words_list = self.get_items_list()
        #     load_all_words_num_before = len(loaded_words_list)
        #     load_all_words_lower_num = 0
        #     load_all_words_upper_num = 0
        #     load_all_words_capitalize_num = 0
        #     for emb_word, emb_vec in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn, emb_delimiter,                                                                                        verbose=True):
        #         if emb_word in loaded_words_list:
        #             continue
        #         if emb_word.lower() not in loaded_words_list and emb_word.lower() not in embeddings_words_list:
        #             self.add_word_emb_vec(emb_word.lower(), emb_vec)
        #             load_all_words_lower_num += 1
        #         if emb_word.upper() not in loaded_words_list and emb_word.upper() not in embeddings_words_list:
        #             self.add_word_emb_vec(emb_word.upper(), emb_vec)
        #             load_all_words_upper_num += 1
        #         if emb_word.capitalize() not in loaded_words_list and emb_word.capitalize() not in \
        #                 embeddings_words_list:
        #             self.add_word_emb_vec(emb_word.capitalize(), emb_vec)
        #             load_all_words_capitalize_num += 1
        #         self.add_item(emb_word)
        #         self.add_emb_vector(emb_vec)
        #     load_all_words_num_after = len(self.get_items_list())
        #     if self.verbose:
        #         print(' ++ load_all_words_num_before = %d ' % load_all_words_num_before)
        #         print(' ++ load_all_words_lower_num = %d ' % load_all_words_lower_num)
        #         print(' ++ load_all_words_num_after = %d ' % load_all_words_num_after)

    def extend_vocab_for_preTrained_model(self, vocab_emb_fn):
        unique_words = []
        emb_vecs =[]
        if self.args.if_extend_vocab == True:
            pretrained_vocab_emb_fn = self.args.diffDomain_pretrainedModel_vocabEmb
            if os.path.exists(pretrained_vocab_emb_fn):
                print('load pre-trained word embedding...')
                fpretrained_read = open(pretrained_vocab_emb_fn, 'rb')
                unique_words, emb_vecs = pickle.load(fpretrained_read)
                print('emb_vecs.shape', np.array(emb_vecs).shape)

                for unique_word, emb_vec in zip(unique_words, emb_vecs):
                    # print('len(emb_vec)', len(emb_vec))
                    self.add_word_emb_vec(unique_word, emb_vec)

        # Get the full list of available case-sensitive words from text file with pretrained embeddings
        if os.path.exists(vocab_emb_fn):
            print('load original dataset pre-trained word embedding...')
            fread = open(vocab_emb_fn, 'rb')
            orig_unique_words, orig_emb_vecs = pickle.load(fread)
            print('orig_emb_vecs.shape', np.array(orig_emb_vecs).shape)
            for orig_unique_word, orig_emb_vec in zip(orig_unique_words, orig_emb_vecs):
                if orig_unique_word not in unique_words:
                    # print('len(orig_emb_vec)', len(orig_emb_vec))
                    self.add_word_emb_vec(orig_unique_word, orig_emb_vec)

    def get_pretrained_embeddingsWeights(self):
        model_path = self.args.diffDomain_pretrainedModel_path
        pretrained_model_dic = torch.load(model_path,self.args.gpu).state_dict()
        char_embeddings = []
        word_embeddings = []
        for name, param in pretrained_model_dic.items():
            if 'char_embeddings' in name:
                char_embeddings = param
            if 'word_embeddings' in name:
                word_embeddings = param
        return char_embeddings, word_embeddings

    def get_embeddings_word(self, word, embeddings_word_list):
        if word in embeddings_word_list:
            self.original_words_num += 1
            return word
        elif self.check_for_lowercase and word.lower() in embeddings_word_list:
            self.lowercase_words_num += 1
            return word.lower()
        elif self.zero_digits and re.sub('\d', '0', word) in embeddings_word_list:
            self.zero_digits_replaced_num += 1
            return re.sub('\d', '0', word)
        elif self.check_for_lowercase and self.zero_digits and re.sub('\d', '0', word.lower()) in embeddings_word_list:
            self.zero_digits_replaced_lowercase_num += 1
            return re.sub('\d', '0', word.lower())
        return None

    def add_word_emb_vec(self, word, emb_vec):
        self.add_item(word)
        self.add_emb_vector(emb_vec)

    def get_unique_characters_list(self, verbose=False, init_by_printable_characters=True):
        if init_by_printable_characters:
            unique_characters_set = set(string.printable)
        else:
            unique_characters_set = set()
        if verbose:
            cnt = 0
        for n, word in enumerate(self.get_items_list()):
            len_delta = len(unique_characters_set)
            unique_characters_set = unique_characters_set.union(set(word))
            if verbose and len(unique_characters_set) > len_delta:
                cnt += 1
                print('n = %d/%d (%d) %s' % (n, len(self.get_items_list), cnt, word))
        return list(unique_characters_set)

    def get_random_embedding(self, embedding_dim=100):
        vocab_size = len(self.unique_words_list)
        pretrain_emb = np.empty([len(self.unique_words_list), embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb



    '''def load_items_from_embeddings(self, emb_fn, emb_delimiter, unique_words_list, emb_load_all):
        if emb_load_all:
            self.load_items_from_embeddings_file_all(emb_fn, emb_delimiter)
        self.load_items_from_embeddings_file_and_unique_words_list(emb_fn, emb_delimiter, unique_words_list)

    def load_items_from_embeddings_file_all(self, emb_fn, emb_delimiter):
        # Get the full list of available case-sensitive words from text file with pretrained embeddings
        embeddings_words_list = [emb_word for emb_word, _ in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn,
                                                                                                          emb_delimiter,
                                                                                                          verbose=True)]
        for emb_word, emb_vec in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn, emb_delimiter,verbose=True):
            self.add_item(emb_word)
            self.add_emb_vector(emb_vec)
            self.original_words_num += 1
            if emb_word.capitalize() not in embeddings_words_list:
                self.add_or_replace_word_emb_vec(word=emb_word.capitalize(), emb_vec=emb_vec)
                self.capitalize_word_num += 1
            if emb_word.upper() not in embeddings_words_list:
                self.add_or_replace_word_emb_vec(word=emb_word.upper(), emb_vec=emb_vec)
                self.uppercase_word_num += 1
        print(' ++ original_words_num = %d' % self.original_words_num)
        print(' ++ capitalize_word_num = %d' % self.capitalize_word_num)
        print(' ++ uppercase_word_num = %d' % self.uppercase_word_num)

    def load_items_from_embeddings_file_and_unique_words_list(self, emb_fn, emb_delimiter, unique_words_list):
        # Get the full list of available case-sensitive words from text file with pretrained embeddings
        embeddings_words_list = [emb_word for emb_word, _ in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn,
                                                                                                                emb_delimiter,
                                                                                                                verbose=True)]
        # Create reverse mapping word from the embeddings file -> list of unique words from the dataset
        emb_word_dict2unique_word_list = dict()
        out_of_vocabulary_words_list = list()
        for unique_word in unique_words_list:
            emb_word = self.get_embeddings_word(unique_word, embeddings_words_list)
            if emb_word is None:
                out_of_vocabulary_words_list.append(unique_word)
            else:
                if emb_word not in emb_word_dict2unique_word_list:
                    emb_word_dict2unique_word_list[emb_word] = [unique_word]
                else:
                    emb_word_dict2unique_word_list[emb_word].append(unique_word)
        # Add pretrained embeddings for unique_words
        for emb_word, emb_vec in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn, emb_delimiter,
                                                                                    verbose=True):
            if emb_word in emb_word_dict2unique_word_list:
                for unique_word in emb_word_dict2unique_word_list[emb_word]:
                    self.add_word_emb_vec(unique_word, emb_vec)
        if self.verbose:
            print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
            print('    First 50 OOV words:')
            for i, oov_word in enumerate(out_of_vocabulary_words_list):
                print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                if i > 49:
                    break
            print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
            print(' -- original_words_num = %d' % self.original_words_num)
            print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
            print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
            print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)

    def add_or_replace_word_emb_vec(self, word, emb_vec):
        if self.item_exists(word):
            idx = self.item2idx_dict[word]
            self.embedding_vectors_list[idx] = emb_vec
        else:
            self.add_word_emb_vec(word, emb_vec)'''
