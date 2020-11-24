from __future__ import print_function
from math import ceil, floor
from os.path import isfile
import time
import numpy as np
import torch.nn as nn
import torch
import random
import sys
# sys.path.append(src.classes.report)
from src.classes.report import Report
from src.classes.utils import *
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_datasets_bank import DatasetsBankFactory
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_optimizer import OptimizerFactory
from src.factories.factory_tagger import TaggerFactory
from src.seq_indexers.seq_indexer_tag import SeqIndexerTag
from src.seq_indexers.seq_indexer_word import SeqIndexerWord



if __name__ == "__main__":
	# torch.backends.cudnn.enabled = False

	random_int = '%08d' % (random.randint(0,100000000))
	print('random_int:',random_int)
	word_freq = 5
	corpus_type = 'ckip' # msr, as, pku, ctb, ckip, cityu, ncc, sxu, weibo
	pretrain_emb = 'xipengqiu' # xinchi, yuezhang, xipengqiu

	# the following only exist a True at most...
	emb_random = False # if character is random, then the vocab needs to new build...
	no_bigram = False  # only utilize the unigram character feature as the input, no bigram character...
	utilized_bigram_pretrained=True # the bigram character utilize the pre-trained, no the average of the two character...

	parser = argparse.ArgumentParser(description='Learning tagger using neural networks')
	parser.add_argument('--splitType', default="num_sentence", type=str,
						help='True to use twitter domain specific pre-trained embedding.')
	parser.add_argument('--value', default=7, type=int,
						help='True to use twitter domain specific pre-trained embedding.')

	parser.add_argument('--model_name', default='',
						help='model name is utilized to name the error figure.')
	parser.add_argument('--corpus_type', default=corpus_type,
						help='model name is utilized to name the error figure.')
	parser.add_argument('--pretrain_emb', default=pretrain_emb,
						help='model name is utilized to name the error figure.')
	parser.add_argument("--word_freq", default=word_freq, type=int,
						help="The number of transformer head ...")
	parser.add_argument("--character_random", default=emb_random, type=str2bool,
						choices=['yes', True, 'no (default)', False],
						help="The unigram character is initialize randomly, and the bigram character is the "
							 "average of two unigram character ")
	parser.add_argument('--if_no_bigram', type=str2bool, default=no_bigram,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='The feature is only the window size unigram character, no bigram character ')
	# parser.add_argument('--if-bigram', type=str2bool, default=True,
	# 					choices=['yes', True, 'no (default)', False], nargs='?',
	# 					help='The feature is only the window size unigram character, no bigram character ')
	parser.add_argument('--if_pretrained_bigram', type=str2bool, default=utilized_bigram_pretrained,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use bert embeddings.')
	parser.add_argument('--multi_train_data', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use bert embeddings.')

	if pretrain_emb == 'xinchi':
		if emb_random:
			parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_biwordFreq' + str(
				word_freq) + '_xinchi_unigramRandomInitialize_vocabEmb.txt',
								help='The embedding of ontonote5 vocabulary.')
		elif no_bigram:
			parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type +'_xinchi_vocabIsUnigramCharacter.txt',
								help='The embedding of ontonote5 vocabulary.')
		else:
			parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_biwordFreq' + str(
				word_freq) + '_xinchi_bigram_vocab_emb.txt',
								help='The embedding of ontonote5 vocabulary.')

		parser.add_argument('--emb-fn', default='../embed/cws/xcchen/vec100.txt', help='Path to word embeddings file.')
		parser.add_argument('--emb-dim', type=int, default=100, help='Dimension of word embeddings file.')

		parser.add_argument('--dropout-ratio', '-r', type=float, default=0.5, help='Dropout ratio.')
		parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
		parser.add_argument('--rnn-hidden-dim', type=int, default=200,
							help='Number hidden units in the recurrent layer.')


	elif pretrain_emb == 'yuezhang':
		if utilized_bigram_pretrained:
			parser.add_argument('--vocab-emb-fn',
								default='emb/' + corpus_type + '_biwordFreq' + str(
									word_freq) + '_yuezhang_predTrainedBigram_vocab_emb.txt',
								help='The embedding of ontonote5 vocabulary.')
		else:
			parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type +'_biwordFreq'+str(word_freq) + '_yuezhang_bigram_vocab_emb.txt',
							help='The embedding of ontonote5 vocabulary.')

		parser.add_argument('--emb-fn', default='../embed/cws/yuezhang/gigaword_chn.all.a2b.uni.ite50.vec', help='Path to word embeddings file.')
		parser.add_argument('--bigram_emb_fn', default='../embed/cws/yuezhang/gigaword_chn.all.a2b.bi.ite50.vec',
							help='Path to word embeddings file.')
		parser.add_argument('--emb-dim', type=int, default=50, help='Dimension of word embeddings file.')

		parser.add_argument('--dropout-ratio', '-r', type=float, default=0.5, help='Dropout ratio.')
		parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
		parser.add_argument('--rnn-hidden-dim', type=int, default=50, help='Number hidden units in the recurrent layer.')

	elif pretrain_emb == 'xipengqiu':
		if utilized_bigram_pretrained:
			parser.add_argument('--vocab-emb-fn',
								default='emb/' + corpus_type + '_biwordFreq' + str(
									word_freq) + '_xipengqiu_predTrainedBigram_vocab_emb.txt',
								help='The embedding of ontonote5 vocabulary.')
		else:
			parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type +'_biwordFreq'+str(word_freq) + '_yuezhang_bigram_vocab_emb.txt',
								help='The embedding of ontonote5 vocabulary.')

		parser.add_argument('--emb-fn', default='../embed/cws/ours-wiki/vectors/1grams_t3_m50_corpus.txt', help='Path to word embeddings file.')
		parser.add_argument('--bigram_emb_fn', default='../embed/cws/ours-wiki/vectors/2grams_t3_m50_corpus.txt',
							help='Path to word embeddings file.')
		parser.add_argument('--emb-dim', type=int, default=100, help='Dimension of word embeddings file.')

		parser.add_argument('--dropout-ratio', '-r', type=float, default=0.5, help='Dropout ratio.')
		parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
		parser.add_argument('--rnn-hidden-dim', type=int, default=64, help='Number hidden units in the recurrent layer.')




	if  corpus_type == 'msr':
		# parser.add_argument('--train', default='data/cws_datasets/data_ncc/small_train',
		#                     help='Train data in format defined by --data-io param.')
		# parser.add_argument('--dev', default='data/cws_datasets/data_ncc/small_dev',
		#                     help='Development data in format defined by --data-io param.')
		# parser.add_argument('--test', default='data/cws_datasets/data_ncc/small_test',
		#                     help='Test data in format defined by --data-io param.')

		parser.add_argument('--train', default='data/cws_datasets/data_msr/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_msr/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_msr/test',
							help='Test data in format defined by --data-io param.')

		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.22, help='Dropout ratio.')

	elif corpus_type == 'as':
		parser.add_argument('--train', default='data/cws_datasets/data_as/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_as/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_as/test',
							help='Test data in format defined by --data-io param.')
		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.17, help='Dropout ratio.')

	elif corpus_type == 'pku':
		parser.add_argument('--train', default='data/cws_datasets/data_pku/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_pku/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_pku/test',
							help='Test data in format defined by --data-io param.')
		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.35, help='Dropout ratio.')

	elif corpus_type == 'ctb':
		parser.add_argument('--train', default='data/cws_datasets/data_ctb/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_ctb/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_ctb/test',
							help='Test data in format defined by --data-io param.')
		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.4, help='Dropout ratio.')

	elif corpus_type == 'ckip':
		parser.add_argument('--train', default='data/cws_datasets/data_ckip/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_ckip/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_ckip/test',
							help='Test data in format defined by --data-io param.')
		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.3, help='Dropout ratio.')

	elif corpus_type == 'cityu':
		parser.add_argument('--train', default='data/cws_datasets/data_cityu/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_cityu/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_cityu/test',
							help='Test data in format defined by --data-io param.')
		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.5, help='Dropout ratio.')

	elif corpus_type == 'ncc':
		parser.add_argument('--train', default='data/cws_datasets/data_ncc/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_ncc/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_ncc/test',
							help='Test data in format defined by --data-io param.')

		# parser.add_argument('--train', default='data/cws_datasets/data_ncc/small_train',
		#                     help='Train data in format defined by --data-io param.')
		# parser.add_argument('--dev', default='data/cws_datasets/data_ncc/small_dev',
		#                     help='Development data in format defined by --data-io param.')
		# parser.add_argument('--test', default='data/cws_datasets/data_ncc/small_test',
		#                     help='Test data in format defined by --data-io param.')

		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.35, help='Dropout ratio.')

	elif corpus_type == 'sxu':
		parser.add_argument('--train', default='data/cws_datasets/data_sxu/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_sxu/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_sxu/test',
							help='Test data in format defined by --data-io param.')
		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.5, help='Dropout ratio.')

	elif corpus_type == 'weibo':
		parser.add_argument('--train', default='data/cws_datasets/data_weibo/train',
							help='Train data in format defined by --data-io param.')
		parser.add_argument('--dev', default='data/cws_datasets/data_weibo/dev',
							help='Development data in format defined by --data-io param.')
		parser.add_argument('--test', default='data/cws_datasets/data_weibo/test',
							help='Test data in format defined by --data-io param.')
		# parser.add_argument('--vocab-emb-fn', default='emb/' + corpus_type + '_vocab_emb.txt',
		#                     help='The embedding of ontonote5 vocabulary.')
		parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
							choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
		# parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
		# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.5, help='Dropout ratio.')


	parser.add_argument('-d', '--data-io', choices=['connl-ner-2003', 'connl-pe', 'connl-wd'],
						default='connl-ner-2003', help='Data read/write file format.')
	parser.add_argument('--gpu', type=int, default=0, help='GPU device number, -1  means CPU.')
	parser.add_argument('--model', help='Tagger model.', choices=['BiRNN', 'BiRNNCNN', 'BiRNNCRF', 'BiRNNCNNCRF'],
						default='BiRNNCNN')
	parser.add_argument('--fn-out-dev', default='results/' + corpus_type + 'dev_results_%s.txt' % random_int,
						help='Path to save the word, true label, and prediction label.')
	parser.add_argument('--fn-out-test', default='results/' + corpus_type + 'test_results_%s.txt' % random_int,
						help='Path to save the word, true label, and prediction label.')

	parser.add_argument('--load', '-l', default=None, help='Path to load from the trained model.')
	parser.add_argument('--save', '-s', default='models/' + corpus_type + '_%s_tagger.hdf5' % random_int,
						help='Path to save the trained model.')
	parser.add_argument('--word-seq-indexer', '-w', type=str, default=None,
						help='Load word_seq_indexer object from hdf5 file.')
	parser.add_argument('--epoch-num', '-e',  type=int, default=200, help='Number of epochs.')
	parser.add_argument('--min-epoch-num', '-n', type=int, default=50, help='Minimum number of epochs.')
	parser.add_argument('--patience', '-p', type=int, default=20, help='Patience for early stopping.')

	parser.add_argument('--save-best', type=str2bool, default=True, help = 'Save best on dev model as a final model.',
						nargs='?', choices=['yes', True, 'no (default)', False])
	# parser.add_argument('--dropout-ratio', '-r', type=float, default=0.5, help='Dropout ratio.')
	# parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size, samples.')
	parser.add_argument('--opt', '-o', help='Optimization method.', choices=['sgd', 'adam'], default='adam')

	parser.add_argument('--lr-decay', type=float, default=0.05, help='Learning decay rate.')
	parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Learning momentum rate.')
	parser.add_argument('--clip-grad', type=float, default=5, help='Clipping gradients maximum L2 norm.')
	parser.add_argument('--rnn-type', help='RNN cell units type.', choices=['Vanilla', 'LSTM', 'GRU','SATN','WCNN'], default='LSTM')
	# parser.add_argument('--rnn-hidden-dim', type=int, default=100, help='Number hidden units in the recurrent layer.')
	# parser.add_argument('--emb-fn', default='../embed/cws/xcchen/vec100.txt', help='Path to word embeddings file.')
	# parser.add_argument('--emb-dim', type=int, default=100, help='Dimension of word embeddings file.')
	parser.add_argument('--emb-delimiter', default=' ', help='Delimiter for word embeddings file.')
	parser.add_argument('--emb-load-all', type=str2bool, default=False, help='Load all embeddings to model.', nargs='?',
						choices = ['yes', True, 'no (default)', False])
	parser.add_argument('--freeze-word-embeddings', type=str2bool, default=False,
						help='False to continue training the word embeddings.', nargs='?',
						choices=['yes', True, 'no (default)', False])
	parser.add_argument('--check-for-lowercase', type=str2bool, default=True, help='Read characters caseless.',
						nargs='?', choices=['yes (default)', True, 'no', False])
	parser.add_argument('--char-embeddings-dim', type=int, default=25, help='Char embeddings dim, only for char CNNs.')
	parser.add_argument('--char-cnn_filter-num', type=int, default=30, help='Number of filters in Char CNN.')
	parser.add_argument('--char-window-size', type=int, default=3, help='Convolution1D size.')
	parser.add_argument('--freeze-char-embeddings', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='False to continue training the char embeddings.')
	parser.add_argument('--word-len', type=int, default=20, help='Max length of words in characters for char CNNs.')
	parser.add_argument('--dataset-sort', type=str2bool, default=False, help='Sort sequences by length for training.',
						nargs='?', choices=['yes', True, 'no (default)', False])
	parser.add_argument('--seed-num', type=int, default=42, help='Random seed number, note that 42 is the answer.')
	parser.add_argument('--report-fn', type=str, default='models/' + corpus_type + '_%s_report.txt' % random_int, help='Report filename.')
	parser.add_argument('--cross-folds-num', type=int, default=-1,
						help='Number of folds for cross-validation (optional, for some datasets).')
	parser.add_argument('--cross-fold-id', type=int, default=-1,
						help='Current cross-fold, 1<=cross-fold-id<=cross-folds-num (optional, for some datasets).')
	parser.add_argument('--verbose', type=str2bool, default=True, help='Show additional information.', nargs='?',
						choices=['yes (default)', True, 'no', False])
	parser.add_argument('--wcnn-layer', type=int, default=2, help='The number of cnn layer on word-level')

	# parser.add_argument('--transform_useLstm', type=str2bool, default=True,
	#                     choices=['yes', True, 'no (default)', False], nargs='?',
	#                     help='True to use bert embeddings.')


	parser.add_argument("--bert_max_seq_length",default=128,type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
							 "Sequences longer than this will be truncated, and sequences shorter \n"
							 "than this will be padded.")
	parser.add_argument("--bert_model", default=None, type=str,required=True,
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
							 "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
							 "bert-base-multilingual-cased, bert-base-chinese.")
	parser.add_argument("--do_lower_case",action='store_true',
						help="Set this flag if you are using an uncased model.")
	parser.add_argument('--bert_output_dim', default=768,type=int,
						help='True to use twitter domain specific pre-trained embedding.')
	parser.add_argument("--cache_dir",default="",type=str,
						help="Where do you want to store the pre-trained models downloaded from s3") #emb/bert_model_cache/bert_cach
	parser.add_argument("--local_rank",type=int,default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument("--window_size", type=int, default=5,
						help="chinese word segmentation window size for local feature.")

	parser.add_argument('--use_CRF', type=str2bool, default=True,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use bert embeddings.')
	parser.add_argument('--transformer', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use bert embeddings.')
	parser.add_argument('--transformer_useSentEncode', type=str2bool, default=True,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use bert embeddings.')
	parser.add_argument("--trans_head", default=4, type=int,
						help="The number of transformer head ...")
	# parser.add_argument("--word_freq", default=5, type=int,
	# 					help="The number of transformer head ...")

	parser.add_argument('--if-glove', type=str2bool, default=True,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use glove embeddings')
	parser.add_argument('--if-wordEmbRand', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use bert embeddings.')
	parser.add_argument('--if-twitter-emb', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use twitter domain specific pre-trained embedding.')
	parser.add_argument('--if_lstmChar', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use char embeddings.')
	parser.add_argument('--if_cnnChar', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use char embeddings.')

	parser.add_argument('--if-elmo', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use elmo embeddings.')
	parser.add_argument('--if-bert', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use bert embeddings.')
	parser.add_argument('--if-flair', type=str2bool, default=False,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use flair embeddings.')

	parser.add_argument('--if-elmo_large', type=str2bool, default=True,
						choices=['yes', True, 'no (default)', False], nargs='?',
						help='True to use elmo large model, output_dim=1024')
	elmo_large = True
	elmo_twitter_indomian = False
	parser.add_argument('--options_file', default='../Projects/ELMo_chinese_wiki_model/elmo_2x4096_512_2048cnn_2xhighway_options.json',
						help='elmo options_file.')
	parser.add_argument('--weight_file', default='../Projects/ELMo_chinese_wiki_model/weight.hdf5',
						help='elmo weight_file.')

	# if elmo_large and not elmo_twitter_indomian:
	# 	parser.add_argument('--options_file', default='../Projects/ELMo/elmo_2x4096_512_2048cnn_2xhighway_options.json',
	# 						help='elmo options_file.')
	# 	parser.add_argument('--weight_file', default='../Projects/ELMo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
	# 						help='elmo weight_file.')
	# elif not elmo_large and not elmo_twitter_indomian:
	# 	parser.add_argument('--options_file', default='../Projects/ELMo/elmo_2x1024_128_2048cnn_1xhighway_options.json',
	# 						help='elmo options_file.')
	# 	parser.add_argument('--weight_file', default='../Projects/ELMo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',
	# 						help='elmo weight_file.')
	# elif elmo_twitter_indomian:
	# 	parser.add_argument('--options_file', default='../Projects/ELMo_Tweet_Domain/elmo_2x4096_512_2048cnn_2xhighway_options.json',
	# 						help='elmo options_file.')
	# 	parser.add_argument('--weight_file', default='../Projects/ELMo_Tweet_Domain/weights.hdf5',
	# 						help='elmo weight_file.')

	args = parser.parse_args()
	np.random.seed(args.seed_num)

	torch.manual_seed(args.seed_num)
	if args.use_CRF:
		args.model = 'BiRNNCNNCRF'
	else:
		args.model = 'BiRNNCNN'

	if args.gpu >= 0:
		torch.cuda.set_device(args.gpu)
		torch.cuda.manual_seed(args.seed_num)
	# Load text data as lists of lists of words (sequences) and corresponding list of lists of tags
	data_io = DataIOFactory.create(args)
	word_sequences_train, word_train, tag_sequences_train, word_sequences_dev, word_dev, tag_sequences_dev, \
	word_sequences_test, word_test, tag_sequences_test = data_io.read_train_dev_test(args)
	# DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches
	# input_bert_train, input_bert_dev, input_bert_test = data_io.read_train_dev_test_bert(args)
	# print('input_bert_train[0]',input_bert_train[0])
	datasets_bank = DatasetsBankFactory.create(args)
	# datasets_bank.add_train_sequences(word_sequences_train, tag_sequences_train)
	# datasets_bank.add_dev_sequences(word_sequences_dev, tag_sequences_dev)
	# datasets_bank.add_test_sequences(word_sequences_test, tag_sequences_test)
	datasets_bank.add_train_sequences(word_sequences_train, word_train, tag_sequences_train)
	datasets_bank.add_dev_sequences(word_sequences_dev, word_dev, tag_sequences_dev)
	datasets_bank.add_test_sequences(word_sequences_test, word_test, tag_sequences_test)
	# Word_seq_indexer converts lists of lists of words to lists of lists of integer indices and back
	# if args.word_seq_indexer is not None and isfile(args.word_seq_indexer):
	#     word_seq_indexer = torch.load(args.word_seq_indexer)
	# else:
		# word_seq_indexer = SeqIndexerWord(gpu=args.gpu, check_for_lowercase=args.check_for_lowercase,
		#                                   embeddings_dim=args.emb_dim, verbose=True)

	word_seq_indexer = SeqIndexerWord(args=args, gpu=args.gpu, check_for_lowercase=args.check_for_lowercase,
									  embeddings_dim=args.emb_dim, verbose=True,
									  unique_words_list=datasets_bank.unique_words_list)

	if args.character_random:
		word_seq_indexer.load_chinese_character_embeddings_randomInitialize(vocab_emb_fn=args.vocab_emb_fn,
																		   emb_fn=args.emb_fn,
																		   emb_delimiter=args.emb_delimiter,
																		   emb_load_all=args.emb_load_all,
																		   unique_words_list=datasets_bank.unique_words_list)
	elif not args.if_pretrained_bigram:
		word_seq_indexer.load_chinese_character_embeddings_likeGlove(vocab_emb_fn=args.vocab_emb_fn,
																		   emb_fn=args.emb_fn,
																		   emb_delimiter=args.emb_delimiter,
																		   emb_load_all=args.emb_load_all,
																		   unique_words_list=datasets_bank.unique_words_list)
	else:
		word_seq_indexer.load_chinese_preTrainedCharBigram_embeddings_likeGlove(vocab_emb_fn=args.vocab_emb_fn,
																	 emb_fn=args.emb_fn,
																	 bigram_emb_fn=args.bigram_emb_fn,
																	 emb_delimiter=args.emb_delimiter,
																	 emb_load_all=args.emb_load_all,
																	 unique_words_list=datasets_bank.unique_words_list)

	# if args.word_seq_indexer is not None and not isfile(args.word_seq_indexer):
	#     torch.save(word_seq_indexer, args.word_seq_indexer)
	# Tag_seq_indexer converts lists of lists of tags to lists of lists of integer indices and back
	tag_seq_indexer = SeqIndexerTag(gpu=args.gpu)
	tag_seq_indexer.load_items_from_tag_sequences(tag_sequences_train)
	# Create or load pre-trained tagger
	item2idx_dic = tag_seq_indexer.item2idx_dict

	if args.load is None:
		tagger = TaggerFactory.create(args, word_seq_indexer, tag_seq_indexer, tag_sequences_train)
	else:
		tagger = TaggerFactory.load(args.load, args.gpu)
	# Create evaluator
	evaluator = EvaluatorFactory.create(args)
	# Create optimizer
	optimizer, scheduler = OptimizerFactory.create(args, tagger)
	# Prepare report and temporary variables for "save best" strategy
	report = Report(args.report_fn, args, score_names=('train loss', '%s-train' % args.evaluator,
													   '%s-dev' % args.evaluator, '%s-test' % args.evaluator))
	# Initialize training variables
	iterations_num = floor(datasets_bank.train_data_num / args.batch_size)
	best_dev_score = -1
	best_epoch = -1
	best_test_score = -1
	best_test_msg = 'N\A'
	patience_counter = 0
	print('\nStart training...\n')

	for epoch in range(0, args.epoch_num + 1):
	# for epoch in range(0, 1):
		time_start = time.time()
		loss_sum = 0
		# if epoch > 0:
		if epoch > -1:
			tagger.train()
			if args.lr_decay > 0:
				scheduler.step()
			for i, (word_sequences_train_batch,input_word_train_batch, tag_sequences_train_batch,) in \
					enumerate(datasets_bank.get_train_batches(args.batch_size)):
				# print('word_sequences_train_batch[0]',word_sequences_train_batch[0])
				# print('tag_sequences_train_batch[0]', tag_sequences_train_batch[0])
				tagger.train()
				tagger.zero_grad()
				loss = tagger.get_loss(word_sequences_train_batch,input_word_train_batch,tag_sequences_train_batch)
				loss.backward()
				nn.utils.clip_grad_norm_(tagger.parameters(), args.clip_grad)
				optimizer.step()
				loss_sum += loss.item()
				if i % 1 == 0:
					batch_time = time.time() -time_start
					print('\r-- train epoch %d/%d, train time %.2fs, batch %d/%d (%1.2f%%), loss = %1.2f.' % (epoch, args.epoch_num, batch_time,
																						 i + 1, iterations_num,
																						 ceil(i*100.0/iterations_num),
																						 loss_sum*100 / iterations_num),
																						 end='', flush=True)
		begin_eval =time.time()
		train_score, dev_score, test_score, test_msg = evaluator.get_evaluation_score_train_dev_test(tagger,
																									 datasets_bank,
																									 item2idx_dic,
																									 batch_size=48)
		evaluate_time = time.time() - begin_eval

		print('\n== eval epoch %d/%d, eval time %.2fs, "%s" train / dev / test | %1.2f / %1.2f / %1.2f.' % (epoch, args.epoch_num,
																											evaluate_time,
																											args.evaluator, train_score,
																											dev_score, test_score))
		report.write_epoch_scores(epoch, (loss_sum*100 / iterations_num, train_score, dev_score, test_score))
		# Save curr tagger if required
		# tagger.save('tagger_NER_epoch_%03d.hdf5' % epoch)
		# Early stopping
		if dev_score >= best_dev_score:
			evaluator.write_WordTargetPred(args, args.fn_out_dev, args.fn_out_test, tagger, datasets_bank,
										   batch_size=48)
			best_dev_score = dev_score
			best_test_score = test_score
			best_epoch = epoch
			best_test_msg = test_msg
			patience_counter = 0
			if args.save is not None and args.save_best:
				tagger.save_tagger(args.save)
			print('## [BEST epoch], %d seconds.\n' % (time.time() - time_start))
		else:
			patience_counter += 1
			print('## [no improvement micro-f1 on DEV during the last %d epochs (best_f1_dev=%1.2f), %d seconds].\n' %
																							(patience_counter,
																							 best_dev_score,
																							 (time.time()-time_start)))
		if patience_counter > args.patience and epoch > args.min_epoch_num:
			break
	# Save final trained tagger to disk, if it is not already saved according to "save best"
	if args.save is not None and not args.save_best:
		tagger.save_tagger(args.save)
	# Show and save the final scores
	if args.save_best:
		report.write_final_score('Final eval on test, "save best", best epoch on dev %d, %s, test = %1.2f)' %
								 (best_epoch, args.evaluator, best_test_score))
		report.write_msg(best_test_msg)
		report.write_input_arguments()
		report.write_final_line_score(best_test_score)
	else:
		report.write_final_score('Final eval on test, %s test = %1.2f)' % (args.evaluator, test_score))
		report.write_msg(test_msg)
		report.write_input_arguments()
		report.write_final_line_score(test_score)
	if args.verbose:
		report.make_print()

