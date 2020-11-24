import codecs
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
from collections import Counter
from pytorch_pretrained_bert.tokenization import BertTokenizer

class DataIOConnlNer2003_bert():
    def DataIOConnlNer2003_bert(self, args):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        input_idss_train, input_masks_train, segment_idss_train, label_idss_train, valids_train, label_masks_train = self.convert_examples_to_features(args.corpus_type, args.train, args.bert_max_seq_length, tokenizer)
        input_idss_dev, input_masks_dev, segment_idss_dev, label_idss_dev, valids_dev, label_masks_dev = self.convert_examples_to_features(args.corpus_type, args.train, args.bert_max_seq_length, tokenizer)
        input_idss_test, input_masks_test, segment_idss_test, label_idss_test, valids_test, label_masks_test = self.convert_examples_to_features(args.corpus_type, args.train, args.bert_max_seq_length, tokenizer)
        return input_idss_train, input_masks_train, segment_idss_train, label_idss_train, valids_train, label_masks_train,\
               input_idss_dev, input_masks_dev, segment_idss_dev, label_idss_dev, valids_dev, label_masks_dev, \
               input_idss_test, input_masks_test, segment_idss_test, label_idss_test, valids_test, label_masks_test



    def read_data(self, corpus_type, fn, verbose=True, column_no=-1):
        if corpus_type =='conll03_pos':
            column_no =1

        word_sequences = list()
        tag_sequences = list()
        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        curr_words = list()
        curr_tags = list()
        for k in range(len(lines)):
            line = lines[k].strip()
            if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
                if len(curr_words) > 0:
                    word_sequences.append(curr_words)
                    tag_sequences.append(curr_tags)
                    curr_words = list()
                    curr_tags = list()
                continue
            strings = line.split(' ')
            word = strings[0]
            tag = strings[column_no] # be default, we take the last tag
            curr_words.append(word)
            curr_tags.append(tag)
            if k == len(lines) - 1:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)

        tag_count = Counter(tag_sequences)
        label_list = [tag for tag,ids in tag_count.most_common(100)]

        return word_sequences, tag_sequences,label_list


    def convert_examples_to_features(self,corpus_type,fn, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        word_sequences, tag_sequences,label_list = self.read_data(corpus_type, fn)
        label_map = {label: i for i, label in enumerate(label_list, 1)}

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
                    else:
                        # label_ids.append(label_map['O'])
                        print('++++++++++++++++')
                        print('labels[i]', labels[i])
                        count_not_find += 1

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
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join(
            #         [str(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

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

        print('num labels not find in label_map', count_not_find)
        return input_idss,input_masks,segment_idss,label_idss,valids,label_masks


