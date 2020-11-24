"""provides storing the train/dev/test data subsets and sampling batches from the train dataset"""

import numpy as np
from random import randint
from src.classes.utils import argsort_sequences_by_lens, get_sequences_by_indices
from collections import Counter
class DatasetsBank():
    """DatasetsBank provides storing the train/dev/test data subsets and sampling batches from the train dataset."""
    def __init__(self, args,verbose=True):
        self.verbose = verbose
        self.unique_words_list = list()
        self.args = args

    def __add_to_unique_words_list(self, word_sequences,min_freq=5):
        total_words = []

        for word_seq in word_sequences:
            for win_seq in word_seq:
                total_words+=win_seq
        # total_words = list(set(total_words) )
        # print('len(total_words)',len(total_words))
        word_count = Counter(total_words)
        print('current dataset total words:',len(word_count.most_common()))
        for w,freq in word_count.most_common():
            if w not in self.unique_words_list:
                if len(w)==1:
                    self.unique_words_list.append(w)
                elif len(w)==2 and freq >min_freq:
                    self.unique_words_list.append(w)
        # for word in total_words:
        #     if word not in self.unique_words_list:
        #         self.unique_words_list.append(word)
        if self.verbose:
            print('DatasetsBank: len(unique_words_list) = %d unique words.' % (len(self.unique_words_list)))


        # count_sent =0
        # for word_seq in word_sequences:
        #     count_sent+=1
        #     print('count_sent',count_sent)
        #     for word in word_seq:
        #         if word not in self.unique_words_list:
        #             self.unique_words_list.append(word)
        # if self.verbose:
        #     print('DatasetsBank: len(unique_words_list) = %d unique words.' % (len(self.unique_words_list)))


    def __add_to_unique_words_list_devtest(self, word_sequences,min_freq=5):
        total_words = []

        for word_seq in word_sequences:
            for win_seq in word_seq:
                total_words+=win_seq
        total_words = list(set(total_words) )
        print('len(total_words)',len(total_words))
        word_count = Counter(total_words)
        for w,freq in word_count.most_common():
            if len(w)==1 and w not in self.unique_words_list:
                self.unique_words_list.append(w)
            # elif len(w)==2 and freq >min_freq:
            #     self.unique_words_list.append(w)

        # for word in total_words:
        #     if word not in self.unique_words_list:
        #         self.unique_words_list.append(word)

        if self.verbose:
            print('DatasetsBank: len(unique_words_list) = %d unique words.' % (len(self.unique_words_list)))


    def __add_to_unique_bigram_words_list(self, word_sequences):
        word_sequences.append('<EOS>')
        word_sequences.append('<EOS>')
        word_sequences.insert(0, '<BOS>')
        word_sequences.insert(0, '<BOS>')
        for word_seq in word_sequences:
            for i in range(len(word_seq)-1):
                bigram_word = word_seq[i]+word_seq[i+1]
                if bigram_word not in self.unique_words_list:
                    self.unique_words_list.append(bigram_word)
        if self.verbose:
            print('DatasetsBank: len(unique_words_list) = %d unique words.' % (len(self.unique_words_list)))


    def add_train_sequences(self, word_sequences_train,input_word_train,tag_sequences_train):
        self.train_data_num = len(word_sequences_train)
        self.word_sequences_train = word_sequences_train
        self.tag_sequences_train = tag_sequences_train
        self.input_word_train = input_word_train
        self.__add_to_unique_words_list(word_sequences_train)
        # else:
        #     self.__add_to_unique_words_list(word_sequences_train)
        #     self.__add_to_unique_bigram_words_list(word_sequences_train)

    def add_dev_sequences(self, word_sequences_dev,input_word_dev, tag_sequences_dev):
        self.word_sequences_dev = word_sequences_dev
        self.tag_sequences_dev = tag_sequences_dev
        self.input_word_dev = input_word_dev
        # self.__add_to_unique_words_list_devtest(word_sequences_dev) # if not use the pre-trained bigram feature
        self.__add_to_unique_words_list(word_sequences_dev) # if use the pre-trained bigram feature.
        # if self.args.if_bigram==False:
        #     self.__add_to_unique_words_list(word_sequences_dev)
        # else:
        #     self.__add_to_unique_words_list(word_sequences_dev)
        #     self.__add_to_unique_bigram_words_list(word_sequences_dev)

    def add_test_sequences(self, word_sequences_test,input_word_test, tag_sequences_test):
        self.word_sequences_test = word_sequences_test
        self.tag_sequences_test = tag_sequences_test
        self.input_word_test = input_word_test
        # self.__add_to_unique_words_list_devtest(word_sequences_test) # if not use the pre-trained bigram feature
        self.__add_to_unique_words_list(input_word_test)  # if use the pre-trained bigram feature.

        # if self.args.if_bigram == False:
        #     self.__add_to_unique_words_list(word_sequences_test)
        # else:
        #     self.__add_to_unique_words_list(word_sequences_test)
        #     self.__add_to_unique_bigram_words_list(word_sequences_test)


    def __get_train_batch(self, batch_indices):
        word_sequences_train_batch = [self.word_sequences_train[i] for i in batch_indices]
        input_word_train_batch = [self.input_word_train[i] for i in batch_indices]
        tag_sequences_train_batch = [self.tag_sequences_train[i] for i in batch_indices]
        # input_ids, input_masks, segment_ids, label_ids, valid_ids, label_masks = self.input_bert_train
        # input_ids_batch = [input_ids[i] for i in batch_indices]
        # input_masks_batch = [input_masks[i] for i in batch_indices]
        # segment_ids_batch = [segment_ids[i] for i in batch_indices]
        # label_ids_batch = [label_ids[i] for i in batch_indices]
        # valid_ids_batch = [valid_ids[i] for i in batch_indices]
        # label_masks_batch = [label_masks[i] for i in batch_indices]
        # input_bert_train_batch = [input_ids_batch,input_masks_batch,segment_ids_batch,label_ids_batch,valid_ids_batch,label_masks_batch]
        # input_bert_train_batch = [self.input_bert_train[:][i] for i in batch_indices]

        return word_sequences_train_batch, input_word_train_batch,tag_sequences_train_batch

    def get_train_batches(self, batch_size):
        random_indices = np.random.permutation(np.arange(self.train_data_num))
        for k in range(self.train_data_num // batch_size): # oh yes, we drop the last batch
            batch_indices = random_indices[k:k + batch_size].tolist()
            word_sequences_train_batch,input_word_train_batch, tag_sequences_train_batch = self.__get_train_batch(batch_indices)
            yield word_sequences_train_batch,input_word_train_batch, tag_sequences_train_batch



class DatasetsBank_backup():
    """DatasetsBank provides storing the train/dev/test data subsets and sampling batches from the train dataset."""
    def __init__(self, args,verbose=True):
        self.verbose = verbose
        self.unique_words_list = list()
        self.args = args

    def __add_to_unique_words_list(self, word_sequences):
        for word_seq in word_sequences:
            for word in word_seq:
                if word not in self.unique_words_list:
                    self.unique_words_list.append(word)
        if self.verbose:
            print('DatasetsBank: len(unique_words_list) = %d unique words.' % (len(self.unique_words_list)))

    def add_train_sequences(self, word_sequences_train, tag_sequences_train):
        self.train_data_num = len(word_sequences_train)
        self.word_sequences_train = word_sequences_train
        self.tag_sequences_train = tag_sequences_train
        self.__add_to_unique_words_list(word_sequences_train)

    def add_dev_sequences(self, word_sequences_dev, tag_sequences_dev):
        self.word_sequences_dev = word_sequences_dev
        self.tag_sequences_dev = tag_sequences_dev
        self.__add_to_unique_words_list(word_sequences_dev)

    def add_test_sequences(self, word_sequences_test, tag_sequences_test):
        self.word_sequences_test = word_sequences_test
        self.tag_sequences_test = tag_sequences_test
        self.__add_to_unique_words_list(word_sequences_test)

    def __get_train_batch(self, batch_indices):
        word_sequences_train_batch = [self.word_sequences_train[i] for i in batch_indices]
        tag_sequences_train_batch = [self.tag_sequences_train[i] for i in batch_indices]
        return word_sequences_train_batch, tag_sequences_train_batch

    def get_train_batches(self, batch_size):
        random_indices = np.random.permutation(np.arange(self.train_data_num))
        for k in range(self.train_data_num // batch_size): # oh yes, we drop the last batch
            batch_indices = random_indices[k:k + batch_size].tolist()
            word_sequences_train_batch, tag_sequences_train_batch = self.__get_train_batch(batch_indices)
            yield word_sequences_train_batch, tag_sequences_train_batch


class DatasetsBankSorted():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.unique_words_list = list()

    def __add_to_unique_words_list(self, word_sequences):
        for word_seq in word_sequences:
            for word in word_seq:
                if word not in self.unique_words_list:
                    self.unique_words_list.append(word)
        if self.verbose:
            print('DatasetsBank: len(unique_words_list) = %d unique words.' % (len(self.unique_words_list)))

    def add_train_sequences(self, word_sequences_train, tag_sequences_train):
        sort_indices, _ = argsort_sequences_by_lens(word_sequences_train)
        self.word_sequences_train = get_sequences_by_indices(word_sequences_train, sort_indices)
        self.tag_sequences_train = get_sequences_by_indices(tag_sequences_train, sort_indices)
        self.train_data_num = len(word_sequences_train)
        self.__add_to_unique_words_list(word_sequences_train)

    def add_dev_sequences(self, word_sequences_dev, tag_sequences_dev):
        self.word_sequences_dev = word_sequences_dev
        self.tag_sequences_dev = tag_sequences_dev
        self.__add_to_unique_words_list(word_sequences_dev)

    def add_test_sequences(self, word_sequences_test, tag_sequences_test):
        self.word_sequences_test = word_sequences_test
        self.tag_sequences_test = tag_sequences_test
        self.__add_to_unique_words_list(word_sequences_test)

    def __get_train_batch(self, batch_size, batch_no, rand_seed=0):
        i = batch_no * batch_size + rand_seed
        j = min((batch_no + 1) * batch_size, self.train_data_num + 1) + rand_seed
        return self.word_sequences_train[i:j], self.tag_sequences_train[i:j]

    def get_train_batches(self, batch_size):
        rand_seed = randint(0, batch_size - 1)
        batch_num = self.train_data_num // batch_size
        random_indices = np.random.permutation(np.arange(batch_num - 1)).tolist()
        for k in random_indices:
            yield self.__get_train_batch(batch_size, batch_no=k, rand_seed=rand_seed)


    def __get_test_batch(self, batch_size, batch_no, rand_seed=0):
        i = batch_no * batch_size + rand_seed
        j = min((batch_no + 1) * batch_size, self.train_data_num + 1) + rand_seed
        return self.word_sequences_train[i:j], self.tag_sequences_train[i:j]

    def get_test_batches(self, batch_size):
        batch_num = self.train_data_num // batch_size
        for k in range(batch_num):
            i = batch_num*batch_size
            j = min((batch_num+1)*batch_size,self.train_data_num+1)
            yield self.word_sequences_train[i:j], self.tag_sequences_train[i:j]

    def __get_train_batch_regularized(self, batch_size, rand_batch_size, batch_no):
        i = batch_no * batch_size
        j = min((batch_no + 1) * batch_size, self.train_data_num + 1)
        word_sequences_train_batch = self.word_sequences_train[i:j]
        tag_sequences_train_batch = self.tag_sequences_train[i:j]
        for k in range(rand_batch_size):
            r = randint(0, self.train_data_num)
            word_sequences_train_batch.append(self.word_sequences_train[r])
            tag_sequences_train_batch.append(self.tag_sequences_train[r])
        return word_sequences_train_batch, tag_sequences_train_batch

    def get_train_batches_regularized(self, batch_size):
        batch_num = self.train_data_num // batch_size
        random_indices = np.random.permutation(np.arange(batch_num)).tolist()
        for k in random_indices:
            yield self.__get_train_batch_regularized(batch_size-2, rand_batch_size=2, batch_no=k)
