import codecs
import os
def read_data_newsplit(fn):
    mode = '\t'
    word_sequences = list()
    tag_sequences = list()
    with open(fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    curr_words = list()
    curr_tags = list()

    for k in range(len(lines)):
        if k == 0:
            continue
        line = lines[k].strip()
        strings = line.split(mode)
        if strings[6] == '</s>':  # new sentence or new document
            if len(curr_words) > 0:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
                curr_words = list()
                curr_tags = list()
            continue

        word = strings[6]
        tag = strings[7]  # be default, we take the last tag
        curr_words.append(word)
        curr_tags.append(tag)
        if k == len(lines) - 1:
            word_sequences.append(curr_words)
            tag_sequences.append(curr_tags)

    return word_sequences, tag_sequences

def read_data_wnut(fn):
    mode = '\t'
    word_sequences = list()
    tag_sequences = list()
    with open(fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    curr_words = list()
    curr_tags = list()

    for k in range(len(lines)):
        line = lines[k].strip()
        strings = line.split(mode)
        if len(line) == 0:  # new sentence or new document
            if len(curr_words) > 0:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
                curr_words = list()
                curr_tags = list()
            continue

        word = strings[0]
        tag = strings[1]  # be default, we take the last tag
        curr_words.append(word)
        curr_tags.append(tag)
        if k == len(lines) - 1:
            word_sequences.append(curr_words)
            tag_sequences.append(curr_tags)

    return word_sequences, tag_sequences


def read_data(corpus_type, fn, verbose=True, column_no=-1):
    mode = ' '
    if corpus_type == 'conll03_pos':
        column_no = 1
        mode = ' '
    elif corpus_type == 'wnut16':
        column_no = 1
        mode = '\t'
        # print('mode',mode)
    elif 'onto' in corpus_type or 'note' in corpus_type:
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
        if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
            if len(curr_words) > 0:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
                curr_words = list()
                curr_tags = list()
            continue
        strings = line.split(mode)
        word = strings[0]
        tag = strings[column_no] # be default, we take the last tag
        curr_words.append(word)
        curr_tags.append(tag)
        if k == len(lines) - 1:
            word_sequences.append(curr_words)
            tag_sequences.append(curr_tags)
    if verbose:
        print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
    return word_sequences, tag_sequences


def read_data_bytype( word_sequences, tag_sequences, value):

    word_sequences2 = []
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
        words_new = []
        tags_new = []
        num_len = int(value)
        if len(word_sequences) <int(value):
            num_len = len(word_sequences)
        for j in range(num_len):
            words_new.append(word_sequences[j])
            tags_new.append(tag_sequences[j])
        word_sequences2.append(words_new)
        tag_sequences2.append(tags_new)
        word_sequences = word_sequences[value:]
        tag_sequences = tag_sequences[value:]

    return word_sequences2, tag_sequences2

def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)


def split_long_to_short(long_file , short_file, value):
    text = "\t".join(["char_i", "word_i", "doc_i", "d_w_i", "d_sent_i", "s_word_i", "word", "tag", "dic_tag"]) + '\n'
    # words_long, tags_long = read_data_newsplit(long_file)
    words_long, tags_long = read_data(corpus_type, long_file, verbose=True, column_no=-1)
    words, tags = read_data_bytype(words_long, tags_long, value)
    char_i = 0
    word_i = 0
    doc_i = 0
    d_w_i = 0
    d_sent_i = 0
    s_word_i = 0
    dic_tag = '<blank>'
    for doc_index in range(len(words)):
        for sent_index in range(len(words[doc_index])):
            for word_index in range(len(words[doc_index][sent_index])):
                word = words[doc_index][sent_index][word_index]
                tag = tags[doc_index][sent_index][word_index]
                text += "\t".join(
                    '%s' % k for k in [char_i, word_i, doc_i, d_w_i, d_sent_i, s_word_i, word, tag, dic_tag]) + '\n'
                char_i += len(word) + len(tag)
                word_i += 1
                d_w_i += 1
                s_word_i += 1
            word = '</s>'
            tag = 'O'
            text += "\t".join(
                '%s' % k for k in [char_i, word_i, doc_i, d_w_i, d_sent_i, s_word_i, word, tag, dic_tag]) + '\n'
            char_i += len(word) + len(tag)
            word_i += 1
            d_w_i += 1
            d_sent_i += 1
            s_word_i = 0
        doc_i += 1
        d_w_i = 0
        d_sent_i = 0


    with open(short_file, 'w') as f:
        f.write(text)

def return_value(corpus_type):
    value = 0
    if corpus_type=="conll03":
        value=10
    elif corpus_type=="notebc":
        value = 6
    elif corpus_type=="notebn":
        value = 4
    elif corpus_type=="notemz":
        value = 3
    elif corpus_type=="notewb":
        value = 7
    elif corpus_type=="notenw":
        value = 6
    elif corpus_type=="notetc":
        value = 10
    return value

if __name__ == "__main__":
    # # 每个document包含4个句子
    # split_long_to_short("test.c_w_d_dw_ds_sw_word_ibo_dic","conll03_split4_test.c_w_d_dw_ds_sw_word_ibo_dic",4)
    # split_long_to_short("train.c_w_d_dw_ds_sw_word_ibo_dic","conll03_split4_train.c_w_d_dw_ds_sw_word_ibo_dic",4)
    # split_long_to_short("valid.c_w_d_dw_ds_sw_word_ibo_dic","conll03_split4_valid.c_w_d_dw_ds_sw_word_ibo_dic",4)
    corpus_types = ["conll03","notebn","notebc","notenw","notemz","notewb","notetc"]

    for corpus_type in corpus_types:
        value = return_value(corpus_type)
        print("corpus_type: %s; value: %s"%(corpus_type,value))
        for data_type in ["train","dev","test"]:

            fn_train_long = "data/raw/"+corpus_type+"/"+data_type+".txt"
            store_path = "data/dset/"+corpus_type
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            fn_train_short = store_path +"/"+data_type+"_value"+str(value)+".c_w_d_dw_ds_sw_word_ibo_dic"
            split_long_to_short(fn_train_long, fn_train_short, value)







