"""token-level accuracy evaluator for each class of BOI-like tags"""
from src.evaluators.evaluator_base import EvaluatorBase
from src.data_io.data_io_connl_ner_2003 import DataIOConnlNer2003

class EvaluatorAccuracyTokenLevel(EvaluatorBase):
    """EvaluatorAccuracyTokenLevel is token-level accuracy evaluator for each class of BOI-like tags."""
    def get_evaluation_score(self, targets_tag_sequences, outputs_tag_sequences, word_sequences=None):
        cnt = 0
        match = 0
        for target_seq, output_seq in zip(targets_tag_sequences, outputs_tag_sequences):
            for t, o in zip(target_seq, output_seq):
                cnt += 1
                if t == o:
                    match += 1
        acc = match*100.0/cnt
        msg = '*** Token-level accuracy: %1.2f%% ***' % acc
        return acc, msg

    def write_WordTargetPred(self, args, fn_out_dev, fn_out_test, tagger, datasets_bank, batch_size=-1):
        d_word_sequences = datasets_bank.word_sequences_dev
        d_targets_tag_sequences = datasets_bank.tag_sequences_dev
        d_outputs_tag_sequences = tagger.predict_tags_from_words(d_word_sequences, batch_size)
        d_data_io_connl_2003 = DataIOConnlNer2003()
        d_data_io_connl_2003.write_data(fn_out_dev, d_word_sequences, d_targets_tag_sequences, d_outputs_tag_sequences)

        word_sequences = datasets_bank.word_sequences_test
        targets_tag_sequences = datasets_bank.tag_sequences_test
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences, batch_size)
        data_io_connl_2003 = DataIOConnlNer2003()
        data_io_connl_2003.write_data(fn_out_test, word_sequences, targets_tag_sequences, outputs_tag_sequences)

