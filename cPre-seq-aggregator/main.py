from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
											  BertConfig,
											  BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report,f1_score
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import codecs

import random
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


class Ner(BertForTokenClassification):

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
				attention_mask_label=None):
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		batch_size, max_len, feat_dim = sequence_output.shape
		valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
		for i in range(batch_size):
			jj = -1
			for j in range(max_len):
				if valid_ids[i][j].item() == 1:
					jj += 1
					valid_output[i][jj] = sequence_output[i][j]
		sequence_output = self.dropout(valid_output)
		logits = self.classifier(sequence_output)

		if labels is not None:
			loss_fct = CrossEntropyLoss(ignore_index=0)
			# Only keep active parts of the loss
			attention_mask_label = None
			if attention_mask_label is not None:
				active_loss = attention_mask_label.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)[active_loss]
				active_labels = labels.view(-1)[active_loss]
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			return loss
		else:
			return logits


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.valid_ids = valid_ids
		self.label_mask = label_mask

#
# def readfile(filename):
# 	'''
# 	read file
# 	'''
# 	f = open(filename)
# 	data = []
# 	sentence = []
# 	label = []
# 	for line in f:
# 		if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
# 			if len(sentence) > 0:
# 				data.append((sentence, label))
# 				sentence = []
# 				label = []
# 			continue
# 		splits = line.split(' ')
# 		sentence.append(splits[0])
# 		# label.append(splits[-1][:-1])
# 		label.append(splits[3].strip()) # note
#
# 	if len(sentence) > 0:
# 		data.append((sentence, label))
# 		sentence = []
# 		label = []
# 	return data



def readfile(filename,args):
	'''
	read file
	'''
	data = []
	word_sequences, tag_sequences = read_data_newsplit(args.corpus_type, filename, verbose=True, column_no=-1)
	logger.info('len(word_sequences): %s' %str(len(word_sequences)) )
	logger.info('len(tag_sequences): %s' % str(len(tag_sequences)))
	logger.info('args.splitType: %s'%args.splitType)
	logger.info('args.value: %s'%args.value)
	word_sequences2, tag_sequences2 = split_data_bytype(word_sequences, tag_sequences, splitType=args.splitType, value=args.value)
	for sent,tags in zip(word_sequences2, tag_sequences2):
		data.append((sent,tags) )

	return data

def read_data_newsplit(corpus_type, fn, verbose=True, column_no=-1):
	mode = ' '
	if corpus_type == 'conll03_pos':
		column_no = 1
		mode = ' '
	elif corpus_type == 'wnut16':
		column_no = 1
		mode = '\t'
	# print('mode',mode)
	elif 'note' in corpus_type:
		column_no = 3
		mode = ' '

	word_sequences = list()
	tag_sequences = list()
	with codecs.open(fn, 'r', 'utf-8') as f:
		lines = f.readlines()
	curr_words = list()
	curr_tags = list()
	for k in range(len(lines)):
		line = lines[k].strip()
		if len(line) == 0 or line.startswith('-DOCSTART-'):  # new sentence or new document
			if len(curr_words) > 0:
				word_sequences.append(curr_words)
				tag_sequences.append(curr_tags)
				curr_words = list()
				curr_tags = list()
			continue
		strings = line.split(mode)
		word = strings[0]
		tag = strings[column_no]  # be default, we take the last tag
		curr_words.append(word)
		curr_tags.append(tag)
		if k == len(lines) - 1:
			word_sequences.append(curr_words)
			tag_sequences.append(curr_tags)
	if verbose:
		logger.info('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
	sent_lens = []
	for k in range(len(word_sequences)):
		sent_lens.append(len(word_sequences[k]))
	avg_len = np.mean(sent_lens)
	logger.info('average sentence length: %f' % avg_len)
	return word_sequences, tag_sequences

def split_data_bytype(word_sequences, tag_sequences, splitType="length", value=256):
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
		tag_all = []
		for words, tags in zip(word_sequences, tag_sequences):
			word_all += words
			tag_all += tags
		word_sequences2 = []
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
			tag_sequences2.append(tag_all[:value])
			word_all = word_all[value:]
			tag_all = tag_all[value:]

		return word_sequences2, tag_sequences2

	elif splitType == "num_sentence":
		word_sequences2 = []
		tag_sequences2 = []
		num = len(word_sequences) / int(value)
		remainder = len(word_sequences) % int(value)
		logger.info('num: %f' %num)
		if remainder != 0:
			num = int(num) + 1
		else:
			num = int(num)
		logger.info('num: %d' %num)

		for i in range(num):
			words_split = []
			tags_split = []
			num_len = int(value)
			if len(word_sequences) <int(value):
				num_len = len(word_sequences)
			for j in range(num_len):
				words_split += word_sequences[j]
				tags_split += tag_sequences[j]
			word_sequences2.append(words_split)
			tag_sequences2.append(tags_split)
			word_sequences = word_sequences[value:]
			tag_sequences = tag_sequences[value:]

		return word_sequences2, tag_sequences2

	elif splitType == "adaptive":
		total_word3 = 0
		total_tag3 = 0
		for k in range(len(word_sequences)):
			total_word3 += len(word_sequences[k])
			total_tag3 += len(tag_sequences[k])
		logger.info('total_word3: %d' % total_word3)
		logger.info('total_tag3: %d' % total_tag3)

		word_sequences2, tag_sequences2 = [], []
		word_sequences1, tag_sequences1 = [], []
		num_word =0
		num_sent =0
		for i in range(len(word_sequences)):

			# if len(word_sequences[i])>=value & num_sent ==1:
			# 	word_sequences1 += word_sequences[i]
			# 	tag_sequences1 += tag_sequences[i]
			num_word += len(word_sequences[i])
			num_sent += 1
			if num_word<value:
				word_sequences1+=word_sequences[i]
				tag_sequences1+=tag_sequences[i]

			else:
				if num_sent==1:
					word_sequences1 =word_sequences[i]
					tag_sequences1  =tag_sequences[i]
					word_sequences2.append(word_sequences1)
					tag_sequences2.append(tag_sequences1)
					num_word = 0
					num_sent = 0
					word_sequences1 = []
					tag_sequences1 = []
				else:
					word_sequences2.append(word_sequences1)
					tag_sequences2.append(tag_sequences1)
					num_word =len(word_sequences[i])
					num_sent =1
					word_sequences1 =word_sequences[i]
					tag_sequences1 =tag_sequences[i]
		if len(word_sequences1)!=0:
			word_sequences2.append(word_sequences1)
			tag_sequences2.append(tag_sequences1)

		total_word2 =0
		total_tag2 = 0
		for j in range(len(word_sequences2)):
			total_word2 +=len(word_sequences2[j])
			total_tag2 +=len(tag_sequences2[j])
		logger.info('total_word2: %d' % total_word2)
		logger.info('total_tag2: %d' % total_tag2)


		return word_sequences2, tag_sequences2


	elif type == "num_document":
		print(3)

def get_words_num(word_sequences):
	return sum(len(word_seq) for word_seq in word_sequences)


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir,args):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir,args):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self,args):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file,args, quotechar=None):
		"""Reads a tab separated value file."""
		return readfile(input_file,args)


class NerProcessor(DataProcessor):
	"""Processor for the CoNLL-2003 data set."""



	def get_train_examples(self, data_dir,args):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "train.txt"),args),"train")

	def get_dev_examples(self, data_dir,args):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "dev.txt"),args), "dev")

	def get_test_examples(self, data_dir,args):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "test.txt"),args), "test")

	def get_labels(self,args):
		# word segmentation
		# tags = ["B","M","E","S","[CLS]", "[SEP]"]
		# conll03
		tags = []
		word_sequences, tag_sequences = read_data_newsplit(args.corpus_type, os.path.join(args.data_dir, "train.txt"), verbose=True, column_no=-1)
		words_test, tags_test = read_data_newsplit(args.corpus_type, os.path.join(args.data_dir, "test.txt"),
														   verbose=True, column_no=-1)

		tags_all = []
		for tag_seq in tag_sequences:
			tags_all+=tag_seq
		for tag_seq in tags_test:
			tags_all+=tag_seq
		tag_count = Counter(tags_all)
		for label,count in tag_count.most_common():
			tags.append(label)

		tags.append("[CLS]")
		tags.append("[SEP]")
		logger.info('tags: %s' % tags)


		# tags =''
		# if args.corpus_type=='conll03':
		# 	tags = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
		#
		# elif args.corpus_type=='notebc' or args.corpus_type=='notebn':
		# 	tags = ['O', 'B-PERSON', 'B-GPE', 'I-ORG', 'I-PERSON', 'I-DATE', 'B-ORG', 'B-NORP', 'B-DATE', 'B-CARDINAL',
		# 			'I-GPE', 'I-CARDINAL', 'I-EVENT', 'I-WORK_OF_ART', 'I-TIME', 'B-TIME', 'B-LOC', 'B-ORDINAL', 'I-LOC',
		# 			'I-QUANTITY', 'B-WORK_OF_ART', 'B-QUANTITY', 'B-EVENT', 'I-MONEY', 'I-LAW', 'I-PERCENT', 'I-FAC',
		# 			'B-FAC', 'B-PERCENT', 'B-MONEY', 'I-NORP', 'B-LAW', 'B-LANGUAGE', 'B-PRODUCT', 'I-LANGUAGE',
		# 			'I-PRODUCT', 'I-ORDINAL',"[CLS]", "[SEP]"]
		# elif args.corpus_type == 'notemz':
		# 	tags = ['O', 'I-PERSON', 'B-PERSON', 'B-GPE', 'I-ORG', 'I-DATE', 'B-ORG', 'B-DATE', 'B-CARDINAL', 'B-NORP',
		# 			'I-WORK_OF_ART', 'I-EVENT', 'I-GPE', 'I-FAC', 'B-ORDINAL', 'B-WORK_OF_ART', 'I-MONEY', 'B-LOC',
		# 			'I-CARDINAL', 'B-EVENT', 'I-LOC', 'B-LANGUAGE', 'I-LAW', 'B-MONEY', 'I-PERCENT', 'I-TIME', 'B-FAC',
		# 			'B-PERCENT', 'B-TIME', 'I-QUANTITY', 'I-NORP', 'B-QUANTITY', 'B-LAW', 'B-PRODUCT', 'I-PRODUCT',
		# 			'I-LANGUAGE', 'I-ORDINAL',"[CLS]", "[SEP]"]
		# elif args.corpus_type == 'notewb':
		# 	tags = ['O', 'B-PERSON', 'I-PERSON', 'B-GPE', 'B-NORP', 'I-ORG', 'I-DATE', 'B-ORG', 'B-DATE', 'B-CARDINAL',
		# 			'I-WORK_OF_ART', 'I-GPE', 'I-FAC', 'I-EVENT', 'I-MONEY', 'B-LOC', 'B-FAC', 'B-ORDINAL', 'I-LOC',
		# 			'B-MONEY', 'I-PERCENT', 'B-QUANTITY', 'B-EVENT', 'B-WORK_OF_ART', 'I-LAW', 'I-CARDINAL', 'I-QUANTITY',
		# 			'B-PERCENT', 'I-NORP', 'B-PRODUCT', 'B-TIME', 'I-TIME', 'B-LAW', 'I-PRODUCT', 'B-LANGUAGE',
		# 			'I-ORDINAL',"[CLS]", "[SEP]"]
		# elif args.corpus_type=='wnut16':
		# 	tags = ['O', 'B-person', 'I-other', 'B-loc', 'B-other', 'I-person', 'B-company', 'I-facility', 'B-facility',
		# 			'B-product', 'I-product', 'I-musicartist', 'B-musicartist', 'B-sportsteam', 'I-loc', 'I-movie',
		# 			'I-company', 'B-movie', 'B-tvshow', 'I-tvshow', 'I-sportsteam', "[CLS]", "[SEP]"]

		return tags

	def _create_examples(self, lines, set_type):
		examples = []
		for i, (sentence, label) in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			text_a = ' '.join(sentence).strip()
			text_b = None
			label = label
			# print('len(sentence)',len(sentence))
			# print('len(label)',len(label))
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
	"""Loads a data file into a list of `InputBatch`s."""

	label_map = {label: i for i, label in enumerate(label_list, 1)}

	features = []
	for (ex_index, example) in enumerate(examples):
		textlist = example.text_a.split(' ')
		labellist = example.label
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
		valid.insert(0, 1)
		label_mask.insert(0, 1)
		label_ids.append(label_map["[CLS]"])
		for i, token in enumerate(tokens):
			ntokens.append(token)
			segment_ids.append(0)
			if len(labels) > i:
				label_ids.append(label_map[labels[i]])
		ntokens.append("[SEP]")
		segment_ids.append(0)
		valid.append(1)
		label_mask.append(1)
		label_ids.append(label_map["[SEP]"])
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

		# if ex_index < 1:
		# 	logger.info("*** Example ***")
		# 	logger.info("guid: %s" % (example.guid))
		# 	logger.info("tokens: %s" % " ".join(
		# 		[str(x) for x in tokens]))
		# 	logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
		# 	logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
		# 	logger.info(
		# 		"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
		# logger.info("label: %s (id = %d)" % (example.label, label_ids))

		features.append(
			InputFeatures(input_ids=input_ids,
						  input_mask=input_mask,
						  segment_ids=segment_ids,
						  label_id=label_ids,
						  valid_ids=valid,
						  label_mask=label_mask))
	return features

def print_args(args):
	logger.info('corpus_type: %s' %args.corpus_type)
	logger.info('splitType: %s'% args.splitType)
	logger.info('value: %s'% str(args.value) )
	logger.info('data_dir: %s'% str(args.data_dir) )
	logger.info('bert_model: %s'% args.bert_model)
	logger.info('task_name: %s'% args.task_name)
	logger.info('output_dir: %s'% args.output_dir)
	logger.info('cache_dir: %s'% args.cache_dir)
	logger.info('max_seq_length: %s'% str( args.max_seq_length) )
	logger.info('do_train: %s'% str(args.do_train) )
	logger.info('do_eval: %s'% str(args.do_eval) )
	logger.info('do_lower_case: %s'% args.do_lower_case)
	logger.info('train_batch_size: %s'% str(args.train_batch_size) )
	logger.info('eval_batch_size: %s'% str(args.eval_batch_size) )
	logger.info('learning_rate: %s'% str(args.learning_rate) )
	logger.info('warmup_proportion: %s' % str(args.warmup_proportion))
	logger.info('no_cuda: %s' % str(args.no_cuda))
	logger.info('num_train_epochs: %s'% str(args.num_train_epochs) )
	logger.info('num_patience: %s' % str(args.num_patience))


def evaluate_model(args, model, device, eval_examples, label_list, tokenizer, random_int):
	eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
	logger.info("***** Running evaluation *****")
	logger.info("  Num examples = %d", len(eval_examples))
	logger.info("  Batch size = %d", args.eval_batch_size)
	all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
	all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
	all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
	eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
							  all_lmask_ids)
	# Run prediction for full data
	eval_sampler = SequentialSampler(eval_data)
	eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
	model.eval()
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	y_true = []
	y_pred = []
	label_map = {i: label for i, label in enumerate(label_list, 1)}
	for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(eval_dataloader,
																				 desc="Evaluating"):
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)
		valid_ids = valid_ids.to(device)
		label_ids = label_ids.to(device)
		l_mask = l_mask.to(device)

		with torch.no_grad():
			logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)

		logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
		logits = logits.detach().cpu().numpy()
		label_ids = label_ids.to('cpu').numpy()
		input_mask = input_mask.to('cpu').numpy()

		for i, label in enumerate(label_ids):
			temp_1 = []
			temp_2 = []
			for j, m in enumerate(label):
				# print('i: ', i)
				# print('j: ', j)
				# print('len(label_ids[i])', len(label_ids[i]))
				# print('label_ids[i][j]',label_ids[i][j])
				# print('label_map: ',label_map)
				# print()
				if j == 0:
					continue
				elif label_ids[i][j] == len(label_map):
					y_true.append(temp_1)
					y_pred.append(temp_2)
					break
				else:
					temp_1.append(label_map[label_ids[i][j]])
					temp_2.append(label_map[logits[i][j]])

	# report = classification_report(y_true, y_pred, digits=4)
	# f1_ss = f1_score(y_true, y_pred)
	# print('f1_ss:',f1_ss)
	# print()
	# f1_s = f1_score(y_true, y_pred, average='micro')
	# print('f1_score: ', f1_s)
	#
	# logger.info("\n%s", report)
	# output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
	# with open(output_eval_file, "w") as writer:
	# 	logger.info("***** Eval results *****")
	# 	logger.info("\n%s", report)
	# 	writer.write(report)

	fn_result_write = 'results/' + str(args.corpus_type) + '_bert_WTP_' + str(random_int) + '.txt'
	fr_write = open(fn_result_write, 'w')
	for (ex_index, example) in enumerate(eval_examples):
		textlist = example.text_a.split(' ')
		leng = len(textlist)
		if len(textlist) >= args.max_seq_length:
			leng = args.max_seq_length - 2
		for idx in range(leng):
			pred_tag = y_pred[ex_index][idx]
			if pred_tag == '[SEP]' or pred_tag == '[CLS]':
				pred_tag = 'O'
			fr_write.write('%s %s %s\n' % (
				textlist[idx], y_true[ex_index][idx], pred_tag))
		fr_write.write('\n')
	fr_write.close()

	if args.evaluator == 'acc':
		score, msg = accuracy_evaluate(fn_result_write)
	else:
		score, msg = standford_ner_evaluate(fn_result_write)

	return score, msg


def accuracy_evaluate(fn_result_write):
	fread = open(fn_result_write, 'r')
	words = []
	y_trues = []
	y_preds = []
	for line in fread.readline():
		if line.strip()!='':
			if len(line.split())>=3:
				print('line:',line)
				wtp = line.split()
				words.append(wtp[0])
				y_trues.append(wtp[1])
				y_preds.append(wtp[2])

	cnt = 0
	match = 0
	for t, o in zip(y_trues, y_preds):
		cnt += 1
		if t == o:
			match += 1
	acc = match * 100.0 / cnt
	msg = '*** Token-level accuracy: %1.2f%% ***' % acc
	return acc, msg


def standford_ner_evaluate(fn_result_write):
	cmd = 'perl %s < %s' % (os.path.join('.', 'conlleval'), fn_result_write)
	msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
	msg += ''.join(os.popen(cmd).readlines())
	# print('msg:')
	# print(msg)
	f1 = float(msg.split('\n')[3].split(':')[-1].strip())

	return f1, msg


def deleteBertFlag(fn_result, fn_rewrite):
	fread = open(fn_result, 'r')
	fwrite = open(fn_rewrite, 'w')
	for line in fread:
		if line.strip() != '':
			pred_tag = line.split()[-1]
			if pred_tag == '[SEP]' or pred_tag == '[CLS]':
				pred_tag = 'O'
			word = line.split()[0]
			true_tag = line.split()[1]

			fwrite.write('%s %s %s \n' % (
				word, true_tag, pred_tag))
		else:
			fwrite.write('\n')


def main():
	random_int = '%08d' % (random.randint(0, 100000000))
	logger.info('random_int: %s'%random_int)

	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument('--corpus_type', type=str, default='conll03', help="Can be used for distant debugging.")
	parser.add_argument('--splitType', default="adaptive_less510", type=str, # num_sentence
						help='True to use twitter domain specific pre-trained embedding.')
	parser.add_argument('--value', default=510, type=int,
						help='True to use twitter domain specific pre-trained embedding.')
	parser.add_argument('--evaluator', default='f1', type=str, choices=['f1','acc'],
						help='True to use twitter domain specific pre-trained embedding.')

	parser.add_argument("--data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--bert_model", default='bert-large-cased', type=str, #required=True,
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
							 "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
							 "bert-base-multilingual-cased, bert-base-chinese.")
	parser.add_argument("--task_name",
						default=None,
						type=str,
						required=True,
						help="The name of the task to train.")
	parser.add_argument("--output_dir",
						default=None,
						type=str,
						required=True,
						help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--result_dir",
						default=None,
						type=str,
						# required=True,
						help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--trainedCorpus",
						default=None,
						type=str,
						# required=True,
						help="The output directory where the model predictions and checkpoints will be written.")
	## Other parameters
	parser.add_argument("--cache_dir",
						default="",
						type=str,
						help="Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--max_seq_length",
						default=512,
						type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
							 "Sequences longer than this will be truncated, and sequences shorter \n"
							 "than this will be padded.")
	parser.add_argument("--do_train",
						action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval",
						action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--do_lower_case",
						action='store_true',
						help="Set this flag if you are using an uncased model.")
	parser.add_argument("--train_batch_size",
						default=2,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=2,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
						default=5e-5,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
						default=10,
						type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--num_patience",
						default=20,
						type=float,
						help="Total number of training epochs to perform.")

	parser.add_argument("--warmup_proportion",
						default=0.1,
						type=float,
						help="Proportion of training to perform linear learning rate warmup for. "
							 "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16',
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
						type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
	parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

	args = parser.parse_args()

	print_args(args)
	print()
	print()

	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd
		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	processors = {"ner": NerProcessor}

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
		device, n_gpu, bool(args.local_rank != -1), args.fp16))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
			args.gradient_accumulation_steps))

	args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	# if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
	# 	raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	task_name = args.task_name.lower()

	if task_name not in processors:
		raise ValueError("Task not found: %s" % (task_name))

	processor = processors[task_name]()
	label_list = processor.get_labels(args)
	num_labels = len(label_list) + 1

	tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

	train_examples = None
	num_train_optimization_steps = None
	if args.do_train:
		train_examples = processor.get_train_examples(args.data_dir,args)
		num_train_optimization_steps = int(
			len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
		if args.local_rank != -1:
			num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

	# Prepare model
	cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
																   'distributed_{}'.format(args.local_rank))
	model = Ner.from_pretrained(args.bert_model,
								cache_dir=cache_dir,
								num_labels=num_labels)
	if args.fp16:
		model.half()
	model.to(device)
	if args.local_rank != -1:
		try:
			from apex.parallel import DistributedDataParallel as DDP
		except ImportError:
			raise ImportError(
				"Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		model = DDP(model)
	elif n_gpu > 1:
		model = torch.nn.DataParallel(model)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	if args.fp16:
		try:
			from apex.optimizers import FP16_Optimizer
			from apex.optimizers import FusedAdam
		except ImportError:
			raise ImportError(
				"Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		optimizer = FusedAdam(optimizer_grouped_parameters,
							  lr=args.learning_rate,
							  bias_correction=False,
							  max_grad_norm=1.0)
		if args.loss_scale == 0:
			optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
		else:
			optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

	else:
		optimizer = BertAdam(optimizer_grouped_parameters,
							 lr=args.learning_rate,
							 warmup=args.warmup_proportion,
							 t_total=num_train_optimization_steps)

	global_step = 0
	nb_tr_steps = 0
	tr_loss = 0
	label_map = {i: label for i, label in enumerate(label_list, 1)}
	best_dev_f1 = 0.0
	patience_counter=0
	if args.do_train:
		train_features = convert_examples_to_features(
			train_examples, label_list, args.max_seq_length, tokenizer)
		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", len(train_examples))
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num steps = %d", num_train_optimization_steps)
		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
		all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
		all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
		train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
								   all_lmask_ids)
		if args.local_rank == -1:
			train_sampler = RandomSampler(train_data)
		else:
			train_sampler = DistributedSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

		model.train()
		# for num_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
		for num_epoch in range(int(args.num_train_epochs)):
			tr_loss = 0
			nb_tr_examples, nb_tr_steps = 0, 0
			for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
			# for step, batch in enumerate(train_dataloader):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
				loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
				if n_gpu > 1:
					loss = loss.mean()  # mean() to average on multi-gpu.
				if args.gradient_accumulation_steps > 1:
					loss = loss / args.gradient_accumulation_steps

				if args.fp16:
					optimizer.backward(loss)
				else:
					loss.backward()

				tr_loss += loss.item()
				nb_tr_examples += input_ids.size(0)
				nb_tr_steps += 1
				if (step + 1) % args.gradient_accumulation_steps == 0:
					if args.fp16:
						# modify learning rate with special warm up BERT uses
						# if args.fp16 is False, BertAdam is used that handles this automatically
						lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
																		  args.warmup_proportion)
						for param_group in optimizer.param_groups:
							param_group['lr'] = lr_this_step
					optimizer.step()
					optimizer.zero_grad()
					global_step += 1

			# begin{save the best model on dev data...}
			dev_examples = processor.get_dev_examples(args.data_dir, args)
			f1, msg = evaluate_model(args, model, device, dev_examples, label_list, tokenizer, random_int)
			if args.evaluator=='acc':
				logger.info('epoch: %s, acc_score: %s, best acc_score: %s' % (str(num_epoch), str(f1), str(best_dev_f1) ))
			else:
				logger.info('epoch: %s, f1_score: %s, best f1_score: %s' % (str(num_epoch), str(f1), str(best_dev_f1) ))
			if f1>best_dev_f1: # save the model...
				best_dev_f1 = f1
				patience_counter = 0
				output_dev_file = os.path.join(args.output_dir, "best_dev_results.txt")
				with open(output_dev_file, "w") as writer:
					writer.write(msg)

				# Save a trained model and the associated configuration
				model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
				output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
				torch.save(model_to_save.state_dict(), output_model_file)
				output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
				with open(output_config_file, 'w') as f:
					f.write(model_to_save.config.to_json_string())
				label_map = {i: label for i, label in enumerate(label_list, 1)}
				model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
								"max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
								"label_map": label_map}
				json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
			else:
				patience_counter+=1
				if args.evaluator=='acc':
					logger.info('no improvement on DEV during the last %d epochs (best_acc_dev=%1.2f),'%(patience_counter,best_dev_f1) )
				else:
					logger.info('no improvement on DEV during the last %d epochs (best_f1_dev=%1.2f),'%(patience_counter,best_dev_f1) )
			if patience_counter>args.num_patience:
				break

			# end{save the best model on dev data...}

		# # Save a trained model and the associated configuration
		# model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
		# output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
		# torch.save(model_to_save.state_dict(), output_model_file)
		# output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
		# with open(output_config_file, 'w') as f:
		# 	f.write(model_to_save.config.to_json_string())
		# label_map = {i: label for i, label in enumerate(label_list, 1)}
		# model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
		# 				"max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
		# 				"label_map": label_map}
		# json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))




	# Load a trained model and config that you have fine-tuned
	else:
		output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
		output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
		config = BertConfig(output_config_file)
		model = Ner(config, num_labels=num_labels)
		model.load_state_dict(torch.load(output_model_file))

	model.to(device)

	if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
		output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
		config = BertConfig(output_config_file)
		model = Ner(config, num_labels=num_labels)
		model.load_state_dict(torch.load(output_model_file))
		model.to(device)

		eval_examples = processor.get_test_examples(args.data_dir, args)
		f1, msg = evaluate_model(args,model,device,eval_examples,label_list,tokenizer,random_int)
		logger.info('test phrase...')
		logger.info(msg)
		output_eval_file = os.path.join(args.output_dir, "test_results.txt")
		with open(output_eval_file, "w") as writer:
			writer.write(msg)


if __name__ == '__main__':
	main()
	# fn_result ='results/conll03Ner_bert_wordTagPred.txt'
	# fn_rewrite = 'results/conll03_CbertWnone_SnoneMlp.txt'
	# deleteBertFlag(fn_result, fn_rewrite)