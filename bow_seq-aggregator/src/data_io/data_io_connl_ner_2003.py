"""input/output data wrapper for CoNNL file format used in  NER-2003 Shared Task dataset"""
import torch
import codecs
from src.classes.utils import get_words_num
from collections import Counter
from pytorch_pretrained_bert.tokenization import BertTokenizer
from copy import deepcopy
class DataIOConnlNer2003():
    """DataIONerConnl2003 is an input/output data wrapper for CoNNL-2003 Shared Task file format.
    Tjong Kim Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003 shared task: Language-independent
    named entity recognition." Proceedings of the seventh conference on Natural language learning at HLT-NAACL
    2003-Volume 4. Association for Computational Linguistics, 2003.
    """
    # def read_train_dev_test(self, args):
    #     if args.multi_train_data:
    #         word_sequences_train, word_train, tag_sequences_train = self.read_data_train(args,args.corpus_type,fns=args.train_fns, verbose=args.verbose)
    #     else:
    #         word_sequences_train,word_train, tag_sequences_train = self.read_data(args,args.corpus_type,fn=args.train, verbose=args.verbose)
    #     word_sequences_dev,word_dev, tag_sequences_dev = self.read_data(args,args.corpus_type,fn=args.dev, verbose=args.verbose)
    #     word_sequences_test,word_test, tag_sequences_test = self.read_data(args,args.corpus_type,fn=args.test, verbose=args.verbose)
    #     return word_sequences_train,word_train, tag_sequences_train, word_sequences_dev,word_dev, tag_sequences_dev, \
    #            word_sequences_test, word_test, tag_sequences_test

    # def read_train_dev(self, args):
    #     word_sequences_train,word_train, tag_sequences_train = self.read_data(args,args.corpus_type,fn=args.train, verbose=args.verbose)
    #     word_sequences_dev,word_dev, tag_sequences_dev = self.read_data(args,args.corpus_type,fn=args.dev, verbose=args.verbose)
    #     # word_sequences_test,word_test, tag_sequences_test = self.read_data(args,args.corpus_type,fn=args.test, verbose=args.verbose)
    #     return word_sequences_train,word_train, tag_sequences_train, word_sequences_dev,word_dev, tag_sequences_dev,
    #     # word_sequences_test, word_test, tag_sequences_test
    #
    # def read_test(self, args):
    #     # word_sequences_train,word_train, tag_sequences_train = self.read_data(args,args.corpus_type,fn=args.train, verbose=args.verbose)
    #     # word_sequences_dev,word_dev, tag_sequences_dev = self.read_data(args,args.corpus_type,fn=args.dev, verbose=args.verbose)
    #     word_sequences_test,word_test, tag_sequences_test = self.read_data(args,args.corpus_type,fn=args.test, verbose=args.verbose)
    #     return word_sequences_test, word_test, tag_sequences_test


    ############################################

    def read_train_dev_test(self, args):
        # if args.multi_train_data:
        #     print('multi_train_data no  yet  ')
        #     # word_sequences_train, word_train, tag_sequences_train = self.read_data_train(args,args.corpus_type,fns=args.train_fns, verbose=args.verbose)
        # else:
        word_sequences_train,word_train, tag_sequences_train = self.read_data(args,args.corpus_type,fn=args.train, verbose=args.verbose)
        word_sequences_dev,word_dev, tag_sequences_dev = self.read_data(args,args.corpus_type,fn=args.dev, verbose=args.verbose)
        word_sequences_test,word_test, tag_sequences_test = self.read_data(args,args.corpus_type,fn=args.test, verbose=args.verbose)

        word_sequences_train2, word_train2, tag_sequences_train2=self.split_data_bytype(args, word_sequences_train,
                                                                                        word_train, tag_sequences_train,
                                                                                        splitType=args.splitType,
                                                                                        value=args.value)
        word_sequences_dev2, word_dev2, tag_sequences_dev2 = self.split_data_bytype(args,
                                                                                          word_sequences_dev,
                                                                                          word_dev,
                                                                                          tag_sequences_dev,
                                                                                            splitType=args.splitType,
                                                                                            value=args.value)
        word_sequences_test2, word_test2, tag_sequences_test2 = self.split_data_bytype(args, word_sequences_test,
                                                                                          word_test,
                                                                                          tag_sequences_test,
                                                                                           splitType=args.splitType,
                                                                                           value=args.value)

        return word_sequences_train2, word_train2, tag_sequences_train2, word_sequences_dev2,word_dev2, tag_sequences_dev2, \
               word_sequences_test2, word_test2, tag_sequences_test2


    def read_data(self,args, corpus_type, fn, verbose=True, column_no=-1):
        mode = '#'
        column_no =-1

        src_data = []
        data = []
        label = []

        src_data_sentence = []
        data_sentence = []
        label_sentence = []

        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        for line in lines:
            # print('line',line)
        # for k in range(len(lines)):
        #     line = str(line, 'utf-8')
            line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
            if len(line_t) < 3:
                if len(data_sentence) == 0:
                    continue
                src_data.append(src_data_sentence)
                data.append(data_sentence)
                label.append(label_sentence)
                src_data_sentence = []
                data_sentence = []
                label_sentence = []
                continue
            src_word = line_t[0]
            word = line_t[1]
            src_data_sentence.append(src_word)
            data_sentence.append(word)
            label_sentence += [line_t[2].split('_')[0]]
        if verbose:
            print('Loading from %s: %d samples, %d words.' % (fn, len(data), get_words_num(data)))
        datas = deepcopy(data)

        # convert the word to window size words ...
        window_datas = []
        for sent in data:
            if not args.if_no_bigram:
                window_datas.append(self.convert2window_bigram_feature(sent, win_size=args.window_size))
            else:
                window_datas.append(self.convert2window_noBigram_feature(sent, win_size=args.window_size))

        # if args.if_bigram:
        #     for sent in data:
        #         window_datas.append(self.convert2window_bigram_feature(sent, win_size=args.window_size) )
        # else:
        #     for sent in data:
        #         window_datas.append(self.convert2window_noBigram_feature(sent, win_size=args.window_size) )

        return window_datas, datas, label



    def split_data_bytype(self, args, word_sequences,unigram_word_sequences, tag_sequences, splitType="length", value=256):
        '''
		:param type:  type is to define the split document-aware data.
					  the data can be split by
						 1) length (such as 512 words,)
						 2) num of sentence (such as 10 sentences)
						 3) num of document
						 4) other? (need to be define)
		:return:
		'''
        if splitType == "length":
            word_all = []
            uws_all = []
            tag_all = []
            for words,uws, tags in zip(word_sequences, unigram_word_sequences,tag_sequences):
                word_all += words
                uws_all += uws
                tag_all += tags
            word_sequences2 = []
            uws_sequences2 = []
            tag_sequences2 = []
            num = len(word_all) / int(value)
            remainder = len(word_all) % int(value)
            # print('num:', num)
            if remainder != 0:
                num = int(num) + 1
            else:
                num = int(num)
            # print('num:', num)

            for i in range(num):
                word_sequences2.append(word_all[:value])
                uws_sequences2.append(uws_all[:value])
                tag_sequences2.append(tag_all[:value])

                word_all = word_all[value:]
                uws_all = uws_all[value:]
                tag_all = tag_all[value:]

            return word_sequences2,uws_sequences2, tag_sequences2

        elif splitType == "num_sentence":
            word_sequences2 = []
            uws_sequences2 = []
            tag_sequences2 = []
            num = len(word_sequences) / int(value)
            remainder = len(word_sequences) % int(value)
            # print('num:', num)
            if remainder != 0:
                num = int(num) + 1
            else:
                num = int(num)
            # print('num:', num)

            for i in range(num):
                words_split = []
                uws_split = []
                tags_split = []
                num_len = int(value)
                if len(word_sequences) < int(value):
                    num_len = len(word_sequences)
                for j in range(num_len):
                    words_split += word_sequences[j]
                    uws_split += unigram_word_sequences[j]
                    tags_split += tag_sequences[j]
                word_sequences2.append(words_split)
                uws_sequences2.append(uws_split)
                tag_sequences2.append(tags_split)
                word_sequences = word_sequences[value:]
                unigram_word_sequences = unigram_word_sequences[value:]
                tag_sequences = tag_sequences[value:]

            return word_sequences2, uws_sequences2,tag_sequences2

        # not need in no-bert-model
        elif splitType == "K-maxSentence":
            total_word3 = 0
            total_tag3 = 0
            for k in range(len(word_sequences)):
                total_word3 += len(word_sequences[k])
                total_tag3 += len(tag_sequences[k])
            print('total_word3: %d' % total_word3)
            print('total_tag3: %d' % total_tag3)

            word_sequences2 = []
            tag_sequences2 = []
            num = len(word_sequences) / int(value)
            remainder = len(word_sequences) % int(value)
            print('num: %f' % num)
            if remainder != 0:
                num = int(num) + 1
            else:
                num = int(num)
            print('num: %d' % num)

            mid_value = int(value / 2)
            print('mid_value:', mid_value)
            while 1:
                if len(word_sequences) == 0:
                    break

                words_split = []
                tags_split = []

                words_split_list = []
                tags_split_list = []
                lens_split = []
                num_len = int(value)
                if len(word_sequences) < int(value):
                    num_len = len(word_sequences)
                for j in range(num_len):
                    # print('word_sequences[j]',word_sequences[j])
                    words_split += word_sequences[j]
                    tags_split += tag_sequences[j]
                    words_split_list.append(word_sequences[j])
                    tags_split_list.append(tag_sequences[j])
                    lens_split.append(len(word_sequences[j]))

                if len(words_split) <= args.max_seq_length:
                    word_sequences2.append(words_split)
                    tag_sequences2.append(tags_split)
                    word_sequences = word_sequences[value:]
                    tag_sequences = tag_sequences[value:]
                else:
                    flag = 0
                    k = 0
                    for k1, w_seq in enumerate(words_split_list):
                        if len(w_seq) > args.max_seq_length:
                            flag = 1
                            k = k1
                            break

                    if flag == 1:
                        # print('k', k)
                        word_seq_pre = []
                        tag_seq_pre = []
                        large_pre_len = 0
                        for m in range(k):
                            large_pre_len += len(words_split_list[m])
                            word_seq_pre += words_split_list[m]
                            tag_seq_pre += tags_split_list[m]

                        if k == 0:
                            word_sequences2.append(words_split_list[k])
                            tag_sequences2.append(tags_split_list[k])
                        elif 0 < k <= mid_value:
                            if len(word_sequences2) != 0:
                                pre_samp_len = len(word_sequences2[-1])
                                merge_len = pre_samp_len + large_pre_len
                                if merge_len <= args.max_seq_length:
                                    # print('len(word_sequences2[-1]), 1: ', len(word_sequences2[-1]))
                                    # print('len(tag_sequences2[-1]), 1: ', len(tag_sequences2[-1]))
                                    word_sequences2_last = word_sequences2[-1] + word_seq_pre
                                    tag_sequences2_last = tag_sequences2[-1] + tag_seq_pre
                                    word_sequences2[-1] = word_sequences2_last
                                    tag_sequences2[-1] = tag_sequences2_last
                                # print('len(word_sequences2[-1]), 2: ', len(word_sequences2[-1]))
                                # print('len(tag_sequences2[-1]), 2: ', len(tag_sequences2[-1]))
                                else:
                                    word_sequences2.append(word_seq_pre)
                                    tag_sequences2.append(tag_seq_pre)
                            else:
                                # print('first samples....')
                                word_sequences2.append(word_seq_pre)
                                tag_sequences2.append(tag_seq_pre)
                            # print('len(word_sequences2[-1]), 22: ', len(word_sequences2[-1]))
                            # print('len(tag_sequences2[-1]), 22: ', len(tag_sequences2[-1]))
                            word_sequences2.append(words_split_list[k])
                            tag_sequences2.append(tags_split_list[k])
                        else:  # j>mid_value
                            word_sequences2.append(word_seq_pre)
                            tag_sequences2.append(tag_seq_pre)
                            word_sequences2.append(words_split_list[k])
                            tag_sequences2.append(tags_split_list[k])
                        word_sequences = word_sequences[k + 1:]
                        tag_sequences = tag_sequences[k + 1:]
                    else:
                        len_count = 0
                        words_split2 = []
                        tags_split2 = []
                        k3 = 0
                        for k2, w_seq in enumerate(words_split_list):
                            len_count += len(w_seq)
                            if len_count <= args.max_seq_length:
                                words_split2 += word_sequences[k2]
                                tags_split2 += tag_sequences[k2]
                                k3 = k2
                        # print('k3',k3)
                        word_sequences2.append(words_split2)
                        tag_sequences2.append(tags_split2)
                        word_sequences = word_sequences[k3 + 1:]
                        tag_sequences = tag_sequences[k3 + 1:]
            # print('word_sequences',word_sequences)

            # for n in range(len(word_sequences2)):
            # 	print('len(word_sequences2[n]):', len(word_sequences2[n]))
            # 	print('len(tag_sequences2[n]):', len(tag_sequences2[n]))

            total_word2 = 0
            total_tag2 = 0
            for j in range(len(word_sequences2)):
                total_word2 += len(word_sequences2[j])
                total_tag2 += len(tag_sequences2[j])
            print('total_word2: %d' % total_word2)
            print('total_tag2: %d' % total_tag2)

            return word_sequences2, tag_sequences2






    def read_data_train(self,args, corpus_type, fns, verbose=True, column_no=-1):
        print('using multi-training set..')
        print('the trainin sent is:',fns)
        src_data = []
        data = []
        label = []

        src_data_sentence = []
        data_sentence = []
        label_sentence = []

        for fn in fns:
            data_corpus =[]
            with codecs.open(fn, 'r', 'utf-8') as f:
                lines = f.readlines()
            for line in lines:
                # print('line',line)
                # for k in range(len(lines)):
                #     line = str(line, 'utf-8')
                line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
                if len(line_t) < 3:
                    if len(data_sentence) == 0:
                        continue
                    src_data.append(src_data_sentence)
                    data.append(data_sentence)
                    label.append(label_sentence)
                    data_corpus.append(data_sentence)

                    src_data_sentence = []
                    data_sentence = []
                    label_sentence = []
                    continue
                src_word = line_t[0]
                word = line_t[1]
                src_data_sentence.append(src_word)
                data_sentence.append(word)
                label_sentence += [line_t[2].split('_')[0]]
            if verbose:
                print('Loading from %s: %d samples, %d words.' % (fn, len(data_corpus), get_words_num(data_corpus)))
        datas = deepcopy(data)

            # convert the word to window size words ...
        window_datas = []
        for sent in data:
            if not args.if_no_bigram:
                window_datas.append(self.convert2window_bigram_feature(sent, win_size=args.window_size))
            else:
                window_datas.append(self.convert2window_noBigram_feature(sent, win_size=args.window_size))

        print('len(datas)',len(datas))
        print('len(window_datas)',len(window_datas))


        return window_datas, datas, label

    def convert2window_bigram_feature(self,sentence,win_size=5):
        window_datas = []

        num = int(win_size / 2)
        for _ in range(num):
            sentence.append('<EOS>')
            sentence.insert(0,'<BOS>')

        for i in range(num,len(sentence)-num):
            window_data = []
            for j in range(-num,num+1):
                window_data.append(sentence[i+j])
            for j in range(-num,num):
                window_data.append(sentence[i+j]+sentence[i+j+1])
            window_datas.append(window_data)
        # print('sentence',sentence)
        # print('window_data',window_data)
        # print('len(sentence)',len(sentence))
        # print('len(window_datas)', len(window_datas))
        return window_datas

    def convert2window_noBigram_feature(self,sentence,win_size=5):
        window_datas = []

        num = int(win_size / 2)
        for _ in range(num):
            sentence.append('<EOS>')
            sentence.insert(0,'<BOS>')
        for i in range(num,len(sentence)-num):
            window_data = []
            for j in range(-num,num+1):
                window_data.append(sentence[i+j])
            window_datas.append(window_data)

        return window_datas



    def write_data(self, fn, word_sequences, tag_sequences_1, tag_sequences_2):
        text_file = open(fn, mode='w')
        for i, words in enumerate(word_sequences):
            tags_1 = tag_sequences_1[i]
            tags_2 = tag_sequences_2[i]
            for j, word in enumerate(words):
                tag_1 = tags_1[j]
                tag_2 = tags_2[j]
                text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
            text_file.write('\n')
        text_file.close()







    ###################################
    # bert dataset pre-processing
    def read_train_dev_test_bert(self, args):
        # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        tokenizer = BertTokenizer.from_pretrained('/home/jlfu/model/cased_L-12_H-768_A-12/vocab.txt',do_lower_case=args.do_lower_case)

        # input_idss_train, input_masks_train, segment_idss_train, label_idss_train, valids_train, label_masks_train = self.convert_examples_to_features(args.corpus_type, args.train, args.bert_max_seq_length, tokenizer)
        # input_idss_dev, input_masks_dev, segment_idss_dev, label_idss_dev, valids_dev, label_masks_dev = self.convert_examples_to_features(args.corpus_type, args.train, args.bert_max_seq_length, tokenizer)
        # input_idss_test, input_masks_test, segment_idss_test, label_idss_test, valids_test, label_masks_test = self.convert_examples_to_features(args.corpus_type, args.train, args.bert_max_seq_length, tokenizer)
        # return input_idss_train, input_masks_train, segment_idss_train, label_idss_train, valids_train, label_masks_train,\
        #        input_idss_dev, input_masks_dev, segment_idss_dev, label_idss_dev, valids_dev, label_masks_dev, \
        #        input_idss_test, input_masks_test, segment_idss_test, label_idss_test, valids_test, label_masks_test

        input_bert_train = self.convert_examples_to_features(
            args.corpus_type, args.train, args.bert_max_seq_length, tokenizer)
        input_bert_dev = self.convert_examples_to_features(
            args.corpus_type, args.dev, args.bert_max_seq_length, tokenizer)
        input_bert_test = self.convert_examples_to_features(
            args.corpus_type, args.test, args.bert_max_seq_length, tokenizer)
        return input_bert_train,input_bert_dev,input_bert_test

    def read_data_bert(self, corpus_type, fn, verbose=True, column_no=-1):
        mode = '#'
        column_no=-1

        # if corpus_type == 'conll03_pos':
        #     column_no = 1
        #     mode = ' '
        # elif corpus_type == 'wnut16':
        #     column_no = 1
        #     mode = '\t'
        #     # print('mode', mode)
        # elif 'onto' in corpus_type or 'note' in corpus_type:
        #     column_no = 3
        #     mode = ' '
        # elif 'onto' in corpus_type or 'note' in corpus_type:
        #     column_no = 3
        #     mode = ' '

        src_data = []
        data = []
        label = []

        src_data_sentence = []
        data_sentence = []
        label_sentence = []
        tag_sequences_all = []

        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        for line in lines:
            # print('line', line)
            # for k in range(len(lines)):
            #     line = str(line, 'utf-8')
            line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
            if len(line_t) < 3:
                if len(data_sentence) == 0:
                    continue
                src_data.append(src_data_sentence)
                data.append(data_sentence)
                label.append(label_sentence)
                src_data_sentence = []
                data_sentence = []
                label_sentence = []
                continue
            src_word = line_t[0]
            word = line_t[1]
            src_data_sentence.append(src_word)
            data_sentence.append(word)
            label_sentence += [line_t[2].split('_')[0]]
            tag_sequences_all.append(line_t[2].split('_')[0])

        tag_count = Counter(tag_sequences_all)
        label_list = [tag[0] for tag in tag_count.most_common(100)]
        pos_label = label

        # if verbose:
        #     print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
        print('label_list', label_list)
        return data, pos_label,label,label_list

    def convert_examples_to_features(self,corpus_type,fn, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        word_sequences,pos_sequences,tag_sequences,label_list = self.read_data_bert(corpus_type, fn)
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        count_not_find = 0
        input_idss,input_masks,segment_idss,label_idss,valids,label_masks = [],[],[],[],[],[]
        # for (ex_index, example) in enumerate(examples):
        for word_seq, tag_seq in zip(word_sequences, tag_sequences):
            textlist = word_seq
            labellist = tag_seq
            tokens = []
            labels = []
            valid = []
            label_mask = []
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]
            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            # valid.insert(0,1)
            # label_mask.insert(0, 1)
            valid.insert(0, 0)
            label_mask.insert(0, 0)
            # label_ids.append(label_map["[CLS]"])

            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    if labels[i] in label_map:
                        label_ids.append(label_map[labels[i]])
                    # else:
                        # label_ids.append(label_map['O'])
                        # print('++++++++++++++++')
                        # print('labels[i]', labels[i])
                        # count_not_find += 1

            ntokens.append("[SEP]")
            segment_ids.append(0)
            # valid.append(1)
            # label_mask.append(1)
            valid.append(0)
            label_mask.append(0)
            # label_ids.append(label_map["[SEP]"])
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            # if ex_index < 5:
            #     print("*** Example ***")
            #     print("guid: %s" % (example.guid))
            #     print("tokens: %s" % " ".join(
            #         [str(x) for x in tokens]))
            #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     print(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # print("label: %s (id = %d)" % (example.label, label_ids))

            # features.append(
            #     InputFeatures(input_ids=input_ids,
            #                   input_mask=input_mask,
            #                   segment_ids=segment_ids,
            #                   label_id=label_ids,
            #                   valid_ids=valid,
            #                   label_mask=label_mask))
            input_idss.append(input_ids)
            input_masks.append(input_mask)
            segment_idss.append(segment_ids)
            label_idss.append(label_ids)
            valids.append(valid)
            label_masks.append(label_mask)
            # print('input_ids',input_ids)
            # print('input_mask',input_mask)
            # print('segment_idss',segment_idss)


        # print('num labels not find in label_map', count_not_find)
        # return [input_idss,input_masks,segment_idss,label_idss,valids,label_masks]
        return [input_idss,input_masks,segment_idss,label_idss,valids,pos_sequences]