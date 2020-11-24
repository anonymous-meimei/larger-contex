"""abstract base class for all evaluators"""
from cws_evaluation import evaluate_word_PRF

class EvaluatorBase():
    """EvaluatorBase is abstract base class for all evaluators"""
    def get_evaluation_score_train_dev_test(self, tagger, datasets_bank, item2idx_dic,batch_size=-1):
        if batch_size == -1:
            batch_size = tagger.batch_size
        score_train, _ = self.predict_evaluation_score(item2idx_dic=item2idx_dic,
                                                       tagger=tagger,
                                                       word_sequences=datasets_bank.word_sequences_train,
                                                       input_sequences =datasets_bank.input_word_train,
                                                       targets_tag_sequences=datasets_bank.tag_sequences_train,
                                                       batch_size=batch_size)
        score_dev, _ = self.predict_evaluation_score(item2idx_dic=item2idx_dic,
                                                     tagger=tagger,
                                                     word_sequences=datasets_bank.word_sequences_dev,
                                                     input_sequences=datasets_bank.input_word_dev,
                                                     targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                     batch_size=batch_size)
        score_test, msg_test = self.predict_evaluation_score(item2idx_dic=item2idx_dic,
                                                             tagger=tagger,
                                                             word_sequences=datasets_bank.word_sequences_test,
                                                             input_sequences=datasets_bank.input_word_test,
                                                             targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                             batch_size=batch_size)
        return score_train, score_dev, score_test, msg_test

    def get_evaluation_score_test_forCrossDomain(self, tagger, datasets_bank, item2idx_dic,batch_size=-1):
        if batch_size == -1:
            batch_size = tagger.batch_size
        # score_train, _ = self.predict_evaluation_score(item2idx_dic=item2idx_dic,
        #                                                tagger=tagger,
        #                                                word_sequences=datasets_bank.word_sequences_train,
        #                                                input_sequences =datasets_bank.input_word_train,
        #                                                targets_tag_sequences=datasets_bank.tag_sequences_train,
        #                                                batch_size=batch_size)
        # score_dev, _ = self.predict_evaluation_score(item2idx_dic=item2idx_dic,
        #                                              tagger=tagger,
        #                                              word_sequences=datasets_bank.word_sequences_dev,
        #                                              input_sequences=datasets_bank.input_word_dev,
        #                                              targets_tag_sequences=datasets_bank.tag_sequences_dev,
        #                                              batch_size=batch_size)
        P, R, F = self.predict_evaluation_score_forCrossDomain(item2idx_dic=item2idx_dic,
                                                             tagger=tagger,
                                                             word_sequences=datasets_bank.word_sequences_test,
                                                             input_sequences=datasets_bank.input_word_test,
                                                             targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                             batch_size=batch_size)
        return P, R, F

    def predict_evaluation_score(self, item2idx_dic,tagger, word_sequences, input_sequences, targets_tag_sequences, batch_size):
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences,input_sequences,batch_size)
        return self.get_evaluation_score(item2idx_dic,targets_tag_sequences, outputs_tag_sequences, input_sequences)


    def predict_evaluation_score_forCrossDomain(self, item2idx_dic,tagger, word_sequences, input_sequences, targets_tag_sequences, batch_size):
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences,input_sequences,batch_size)
        return self.get_evaluation_score_forCrossDomain(item2idx_dic,targets_tag_sequences, outputs_tag_sequences, input_sequences)


