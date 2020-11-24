"""f1-micro averaging evaluator for tag components, spans detection + classification, uses standard CoNNL perl script"""
from __future__ import division
import os
import random
import time
from src.data_io.data_io_connl_ner_2003 import DataIOConnlNer2003
from src.evaluators.evaluator_base import EvaluatorBase
from collections import Counter
import matplotlib.mlab as mlab
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
# from ner_evaluate import evaluate,evaluate_each_class
from cws_evaluation import evaluate_word_PRF


class EvaluatorF1MicroSpansConnl(EvaluatorBase):
    """EvaluatorF1Connl is f1-micro averaging evaluator for tag components, standard CoNNL perl script."""
    def get_evaluation_score(self, item2idx_dic,targets_tag_sequences, outputs_tag_sequences, input_sequences):
        fn_out = 'out_temp_%04d.txt' % random.randint(0, 10000)
        if os.path.isfile(fn_out):
            os.remove(fn_out)
        data_io_connl_2003 = DataIOConnlNer2003()
        data_io_connl_2003.write_data(fn_out, input_sequences, targets_tag_sequences, outputs_tag_sequences)

        # cmd = 'perl %s < %s' % (os.path.join('.', 'conlleval'), fn_out)
        # msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
        # msg += ''.join(os.popen(cmd).readlines())
        # time.sleep(0.5)
        if fn_out.startswith('out_temp_') and os.path.exists(fn_out):
            os.remove(fn_out)
        # f1 = float(msg.split('\n')[3].split(':')[-1].strip())

        f1 = evaluate_word_PRF(targets_tag_sequences, outputs_tag_sequences, item2idx_dic,test=False)
        msg = 'msg'
        return f1, msg

    def get_evaluation_score_forCrossDomain(self, item2idx_dic,targets_tag_sequences, outputs_tag_sequences, input_sequences):
        fn_out = 'out_temp_%04d.txt' % random.randint(0, 10000)
        if os.path.isfile(fn_out):
            os.remove(fn_out)
        data_io_connl_2003 = DataIOConnlNer2003()
        data_io_connl_2003.write_data(fn_out, input_sequences, targets_tag_sequences, outputs_tag_sequences)

        # cmd = 'perl %s < %s' % (os.path.join('.', 'conlleval'), fn_out)
        # msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
        # msg += ''.join(os.popen(cmd).readlines())
        # time.sleep(0.5)
        if fn_out.startswith('out_temp_') and os.path.exists(fn_out):
            os.remove(fn_out)
        # f1 = float(msg.split('\n')[3].split(':')[-1].strip())

        P, R, F = evaluate_word_PRF(targets_tag_sequences, outputs_tag_sequences, item2idx_dic,test=True)
        msg = 'msg'
        return P, R, F

    def write_WordTargetPred(self, args, fn_out_dev, fn_out_test, tagger, datasets_bank, batch_size=-1):
        d_word_sequences = datasets_bank.word_sequences_dev
        d_input_sequences = datasets_bank.input_word_dev
        d_targets_tag_sequences = datasets_bank.tag_sequences_dev
        d_outputs_tag_sequences = tagger.predict_tags_from_words(d_word_sequences,d_input_sequences,batch_size)
        d_data_io_connl_2003 = DataIOConnlNer2003()
        d_data_io_connl_2003.write_data(fn_out_dev, d_input_sequences, d_targets_tag_sequences, d_outputs_tag_sequences)

        word_sequences = datasets_bank.word_sequences_test
        input_sequences = datasets_bank.input_word_test
        targets_tag_sequences = datasets_bank.tag_sequences_test
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences,input_sequences,batch_size)
        data_io_connl_2003 = DataIOConnlNer2003()
        data_io_connl_2003.write_data(fn_out_test, input_sequences, targets_tag_sequences, outputs_tag_sequences)

    def write_WordTargetPred_forCrossDomain(self, args,fn_out_test, tagger, datasets_bank, batch_size=-1):
        # d_word_sequences = datasets_bank.word_sequences_dev
        # d_input_sequences = datasets_bank.input_word_dev
        # d_targets_tag_sequences = datasets_bank.tag_sequences_dev
        # d_outputs_tag_sequences = tagger.predict_tags_from_words(d_word_sequences, d_input_sequences, batch_size)
        # d_data_io_connl_2003 = DataIOConnlNer2003()
        # d_data_io_connl_2003.write_data(fn_out_dev, d_input_sequences, d_targets_tag_sequences,
        #                                 d_outputs_tag_sequences)

        word_sequences = datasets_bank.word_sequences_test
        input_sequences = datasets_bank.input_word_test
        targets_tag_sequences = datasets_bank.tag_sequences_test
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences, input_sequences, batch_size)
        data_io_connl_2003 = DataIOConnlNer2003()
        data_io_connl_2003.write_data(fn_out_test, input_sequences, targets_tag_sequences, outputs_tag_sequences)

        # acc, f1, p, r, c_f1, c_p, c_r =self.evaluate_score(word_sequences, targets_tag_sequences, outputs_tag_sequences)
        # print('acc:%f, f1:%f, p:%f, r:%f, c_f1:%f, c_p:%f, c_r:%f' %(acc, f1, p, r, c_f1, c_p, c_r))

        # print('\n')
        # acc, f1, p, r = evaluate(outputs_tag_sequences, targets_tag_sequences, word_sequences)
        # print('total// acc:%f, f1:%f, p:%f, r:%f ' %(acc, f1, p, r))
        # class_types = ['PER', 'LOC', 'ORG']
        # if 'ontonote5' in  args.corpus_type:
        #     class_types = ['PERSON', 'LOC', 'ORG']
        # if args.corpus_type == 'wnut16':
        #     class_types = ['person', 'loc', 'company']
        #
        # for class_type in class_types:
        #     c_f1, c_p, c_r = evaluate_each_class(outputs_tag_sequences, targets_tag_sequences, word_sequences, class_type)
        #     print('class_type  %s: f1:%f, p:%f, r:%f' %(class_type,c_f1, c_p, c_r))

    # def write_WordTargetPredProb(self, fn_out_test, tagger, datasets_bank, probs_list, batch_size=-1):
    #     word_sequences = datasets_bank.word_sequences_test
    #     targets_tag_sequences = datasets_bank.tag_sequences_test
    #     outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences, batch_size)
    #     data_io_connl_2003 = DataIOConnlNer2003()
    #     data_io_connl_2003.write_data_prob(fn_out_test, word_sequences, targets_tag_sequences, outputs_tag_sequences, probs_list)

    # def draw_pie_error_statics(self, args, fn_out_test):
    #     corpus_type = ''
    #     if args.corpus_type == 'connl03':
    #         corpus_type = 'CoNLL03'
    #     elif args.corpus_type == 'wnut16':
    #         corpus_type = 'WNUT16'
    #     elif args.corpus_type == 'ontonote5':
    #         corpus_type = 'OntoNotes5.0'
    #     elif args.corpus_type == 'ptbPos':
    #         corpus_type = 'ptbPos'
    #     elif args.corpus_type == 'chunk':
    #         corpus_type = 'chunk'
    #     elif args.corpus_type == 'ontonote5chunk':
    #         corpus_type = 'ontonote5Pos'
    #     elif args.corpus_type == 'ontonote5Pos':
    #         corpus_type = 'ontonote5Pos'
    #
    #     else:
    #         print('corpus_type error...')
    #
    #     # model_name =''
    #     # if args.if_char:
    #     #     model_name+= 'char_cnn '
    #     # elif args.if_glove:
    #     #     model_name += 'word '
    #     # elif args.if_elmo:
    #     #     model_name += 'elmo '
    #     # elif args.if_bert:
    #     #     model_name += 'bert '
    #     model_name = args.model_name
    #
    #     file_name = fn_out_test.split('/')[1][:-4]
    #     words = []
    #     true_labels = []
    #     pred_labels = []
    #     with open(fn_out_test, 'r') as ftest_read:
    #         for line in ftest_read:
    #             if line.strip() != '':
    #                 line_list = line.split()
    #                 words.append(line_list[0])
    #                 true_labels.append(line_list[1])
    #                 pred_labels.append(line_list[2])
    #     label_counts = Counter(true_labels)
    #     vocab_label = [x[0] if x[0] =='O' or x[0] =='o' else x[0].split('-')[1] for x in label_counts.most_common()]
    #     tags = list(set(vocab_label))
    #
    #     # count total error num...
    #     total_error = 0
    #     for i in range(len(true_labels)):
    #         if true_labels[i] != pred_labels[i]:
    #             total_error += 1
    #     print('total error num is: ', total_error)
    #
    #     Y = []
    #     X = []
    #     for i in range(len(tags)):
    #         for j in range(len(tags)):
    #             if not i == j:
    #                 string = tags[i] + '->' + tags[j]
    #                 Y.append(string)
    #                 error_count = 0
    #                 for word, t_tag, p_tag in zip(words, true_labels, pred_labels):
    #                     t_tag = t_tag if t_tag =='O' or t_tag =='o' else t_tag.split('-')[1]
    #                     p_tag = p_tag if p_tag == 'O' or p_tag == 'o' else p_tag.split('-')[1]
    #                     # print('t_tag', t_tag)
    #                     # print('tags[i]', tags[i])
    #                     # print('p_tag', p_tag)
    #                     # print('tags[j]',tags[j])
    #                     # print('\n')
    #                     if t_tag == tags[i] and p_tag == tags[j]:
    #                         error_count += 1
    #                 X.append(error_count)
    #     X_sorted, Y_sorted = (list(t) for t in zip(*sorted(zip(X, Y), reverse=True)))
    #     print('len(X_sorted)', len(X_sorted))
    #     print('len(Y_sorted)', len(Y_sorted))
    #     print('X_sorted', X_sorted)
    #     print('Y_sorted', Y_sorted)
    #
    #     # starting to draw figure...
    #     color_list = ['#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#71AD47', '#264478', '#9E480D', '#636363', '#997300',
    #                   '#255E91', '#43682B', '#698ED0', '#F1975A', '#B7B7B7', '#FFCD32', '#8CC168', '#8CC168']
    #     fig1, ax1 = plt.subplots()
    #     colors = color_list * int(len(X_sorted) / len(color_list) + 1)
    #     colors = colors[:len(X_sorted)]
    #
    #     labels = []
    #     for i in range(len(X_sorted)):
    #         labels.append(str(Y_sorted[i]) + ', ' + str(X_sorted[i]))
    #
    #     patches, texts, autotexts = ax1.pie(X_sorted, labels=labels, colors=colors, labeldistance=1.1,
    #                                         autopct='%1.1f%%', startangle=180, pctdistance=0.9)
    #     # ax1.axis('equal')
    #     plt.title(corpus_type + 'Top total: ' +str(len(X_sorted)) + ' Error Statistic \n' +'Model Name: '+model_name + 'Total_error: ' +str(total_error))
    #
    #     # set foot size.
    #     proptease = fm.FontProperties()
    #     proptease.set_size('x-small')
    #
    #     plt.setp(autotexts, fontproperties=proptease)
    #     plt.setp(texts, fontproperties=proptease)
    #
    #     plt.show()
    #     save_path = 'error_fig/' + corpus_type +' Model Name: '+model_name + '_test_%s.pdf' % file_name
    #     plt.savefig(save_path)
    #     plt.close()
    #
    #     # only draw top 15 type error.
    #     X_top15 = X_sorted[:15]
    #     # print('X_top15', X_top15)
    #     Y_top15 = Y_sorted[:15]
    #     # print('Y_top15', Y_top15)
    #
    #
    #     Top15_scale = float('%.2f' %(sum(X_top15) / total_error))
    #     print('Top15_scale', Top15_scale)
    #     fig2, ax2 = plt.subplots()
    #     labels15 = labels[:15]
    #     patches_top15, texts_top15, autotexts_top15 = ax2.pie(X_top15, labels=labels15, colors=colors, labeldistance=1.1,
    #                                                           autopct='%1.1f%%', startangle=180, pctdistance=0.9)
    #     plt.title(corpus_type + ' Top 15' + ' Error Statistic \n' +'Model Name: '+model_name + 'Total_error: ' +str(total_error)+', Top15_scale: ' +str(Top15_scale))
    #     # set foot size.
    #     proptease_top15 = fm.FontProperties()
    #     proptease_top15.set_size('x-small')
    #     plt.setp(autotexts_top15, fontproperties=proptease_top15)
    #     plt.setp(texts_top15, fontproperties=proptease_top15)
    #
    #     plt.show()
    #     save_path = 'error_fig/' + corpus_type +' Model Name: '+model_name +  '_test_top15_%s.pdf' % file_name
    #     plt.savefig(save_path)
    #     plt.close()
    #
    #
    #     # only draw top 10 type error.
    #     X_top10 = X_sorted[:10]
    #     # print('X_top10', X_top10)
    #     Y_top10 = Y_sorted[:10]
    #     # print('Y_top10', Y_top10)
    #
    #     Top10_scale = float('%.2f' % (sum(X_top10) / total_error))
    #     print('Top10_scale', Top10_scale)
    #     fig2, ax2 = plt.subplots()
    #     labels10 = labels[:10]
    #     patches_top10, texts_top10, autotexts_top10 = ax2.pie(X_top10, labels=labels10, colors=colors, labeldistance=1.1,
    #                                                           autopct='%1.1f%%', startangle=180, pctdistance=0.9)
    #     plt.title(corpus_type + ' Top 10' + ' Error Statistic \n' +'Model Name: '+model_name +  'Total_error: ' + str(
    #         total_error) + ', Top10_scale: ' + str(Top10_scale))
    #     # set foot size.
    #     proptease_top10 = fm.FontProperties()
    #     proptease_top10.set_size('x-small')
    #     plt.setp(autotexts_top10, fontproperties=proptease_top10)
    #     plt.setp(texts_top10, fontproperties=proptease_top10)
    #
    #     plt.show()
    #     save_path = 'error_fig/' + corpus_type +' Model Name: '+model_name + '_test_top10_%s.pdf' % file_name
    #     plt.savefig(save_path)
    #     plt.close()


    # def draw_pie_error_statics1(self, args, fn_out_test):
    #     fn_out_test = 'results/conll03_test_results_2019_03_16_14-17_05.txt'
    #     words = []
    #     true_labels = []
    #     pred_labels = []
    #     with open(fn_out_test, 'r') as ftest_read:
    #         for line in ftest_read:
    #             if line.strip() != '':
    #                 line_list = line.split()
    #                 words.append(line_list[0])
    #                 true_labels.append(line_list[1])
    #                 pred_labels.append(line_list[2])
    #     label_counts = Counter(true_labels)
    #     vocab_label = [x[0] if x[0] =='O' or x[0] =='o' else x[0].split('-')[1] for x in label_counts.most_common()]
    #     # print('vocab_label', vocab_label)
    #     # vocab_label = [x[0].split('e')[1] for x in label_counts.most_common()]
    #     tags = list(set(vocab_label))
    #     # print('tags', tags)
    #
    #
    #     Y = []
    #     X = []
    #     for i in range(len(tags)):
    #         for j in range(len(tags)):
    #             if not i == j:
    #                 string = tags[i] + '->' + tags[j]
    #                 Y.append(string)
    #                 error_count = 0
    #                 for word, t_tag, p_tag in zip(words, true_labels, pred_labels):
    #                     t_tag = t_tag if t_tag =='O' or t_tag =='o' else t_tag.split('-')[1]
    #                     p_tag = p_tag if p_tag == 'O' or p_tag == 'o' else p_tag.split('-')[1]
    #                     # print('t_tag', t_tag)
    #                     # print('tags[i]', tags[i])
    #                     # print('p_tag', p_tag)
    #                     # print('tags[j]',tags[j])
    #                     # print('\n')
    #                     if t_tag == tags[i] and p_tag == tags[j]:
    #                         error_count += 1
    #                 X.append(error_count)
    #     # fig = plt.figure()
    #     print('X', X)
    #     print('Y', Y)
    #     print('len(X)', len(X))
    #     print('len(Y)', len(Y))
    #     plt.pie(X, labels=Y, autopct='%1.2f%%')
    #     plt.title(args.corpus_type + ' Error Statistic')
    #
    #     plt.show()
    #     save_path = 'error_fig/' + args.corpus_type +'_test_%04d.jpg' % random.randint(0, 10000)
    #     plt.savefig(save_path)
    #     plt.close()
    #
    #     # only draw top 15 type error.
    #     # X1=X[:15]
    #     # Y1=Y[:15]
    #     X1, Y1 = (list(t) for t in zip(*sorted(zip(X, Y))))
    #     X1 = X1[:15]
    #     Y1 = Y1[:15]
    #     plt.pie(X1, labels=Y1, autopct='%1.2f%%')
    #     plt.title('Top 15' + args.corpus_type + ' Error Statistic')
    #     plt.show()
    #     save_path = 'error_fig/' + args.corpus_type + '_test_top15_%04d.jpg' % random.randint(0, 10000)
    #     plt.savefig(save_path)
    #     plt.close()




