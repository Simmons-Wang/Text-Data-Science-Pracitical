import os
import csv
import math
from tqdm import tqdm
from stemming.porter2 import stem
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import re
import string
import scipy
from scipy import sparse
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools

os.chdir(r'C:\Users\Simmons\PycharmProjects\ttds')


# %% IR EVALUATION
def my_reader(sys_result_path, qrel_path):
    """
    :param sys_result_path: the path of system_results.csv
    :param qrel_path: the path of qrels.csv
    :return: the dict storing the system_results and the dict storing the qrels
    """
    with open(sys_result_path, 'r') as f:
        system_results = [ row for row in csv.reader(f) ]
    result_dict = {}
    for i in range(1, 7):
        result_dict[ i ] = {}
        for j in range(1, 11):
            result_dict[ i ][ j ] = {'doc_nums': [ ], 'scores': [ ]}
    for row in system_results[ 1: ]:
        line = [ float(i) for i in row ]
        result_dict[ line[ 0 ] ][ line[ 1 ] ][ 'doc_nums' ].append(line[ 2 ])
        result_dict[ line[ 0 ] ][ line[ 1 ] ][ 'scores' ].append(line[ 4 ])

    with open(qrel_path, 'r') as f:
        qrels = [ row for row in csv.reader(f) ]
    qrel_dict = {}
    for i in range(1, 11):
        qrel_dict[ i ] = {'doc_id': [ ], 'relevance': [ ]}
    for row in qrels[ 1: ]:
        line = [ float(i) for i in row ]
        qrel_dict[ line[ 0 ] ][ 'doc_id' ].append(line[ 1 ])
        qrel_dict[ line[ 0 ] ][ 'relevance' ].append(line[ 2 ])

    return result_dict, qrel_dict


def p_k(rel_nums, sys_result, k):
    """calculate the Precision@k"""
    p = len(set(set(rel_nums) & set(sys_result[ :k ]))) / k
    return p


def r_k(rel_nums, sys_result, k):
    """calculate the Recall@k"""
    r = len(set(set(rel_nums) & set(sys_result[ :k ]))) / len(rel_nums)
    return r


def ap_cal(rel_nums, sys_result):
    """calculate the average precision"""
    aps = 0
    for n, id in enumerate(sys_result):
        if id in rel_nums:
            p = p_k(rel_nums, sys_result, n + 1)
            aps += p
    ap = aps / len(rel_nums)
    return ap


def DCG_cal(Gs):
    """calculate the DCG of the last k"""
    DCG = 0
    for n, G in enumerate(Gs):
        if n == 0:
            DG = G
        else:
            DG = G / math.log2(n + 1)
        DCG += DG
    return DCG


def nDCG_cal(rel_nums, rel_scores, sys_result, k):
    """
    :param rel_nums: the list of relevant documents
    :param rel_scores: the relevance scores of  relevant documents
    :param sys_result: the retrieval result of the system
    :param k: k
    :return: nDCG@k
    """
    Gs = [ ]
    for n, id in enumerate(sys_result[ :k ]):
        if id in rel_nums:
            ii = rel_nums.index(id)
            G = rel_scores[ii]
        else:
            G = 0
        Gs.append(G)
    DCG_k = DCG_cal(Gs)
    Gs.sort(reverse=True)
    iDCG_k = DCG_cal(Gs)
    if iDCG_k == 0:
        return 0
    nDCG_k = DCG_k / iDCG_k
    return nDCG_k


result_dict, qrel_dict = my_reader(r'./input/cw2/system_results.csv', r'./input/cw2/qrels.csv')

with open(r'./output/ir_eval.csv', 'w', newline="") as f:
    write = csv.writer(f)
    write.writerow([ 'system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20' ])
    for i in range(1, 7):  # go through six systems
        p_10_sum = 0
        r_50_sum = 0
        r_p_sum = 0
        ap_sum = 0
        nDCG_cal_10_sum = 0
        nDCG_cal_20_sum = 0
        for j in range(1, 11):  # go through the 10 queries
            sys_result = result_dict[ i ][ j ][ 'doc_nums' ]
            rel_nums = qrel_dict[ j ][ 'doc_id' ]
            rel_scores = qrel_dict[ j ][ 'relevance' ]
            p_10 = p_k(rel_nums, sys_result, 10)
            p_10_sum += p_10
            r_50 = r_k(rel_nums, sys_result, 50)
            r_50_sum += r_50
            r_p = r_k(rel_nums, sys_result, len(rel_nums))
            r_p_sum += r_p
            ap = ap_cal(rel_nums, sys_result)
            ap_sum += ap
            nDCG_cal_10 = nDCG_cal(rel_nums, rel_scores, sys_result, 10)
            nDCG_cal_10_sum += nDCG_cal_10
            nDCG_cal_20 = nDCG_cal(rel_nums, rel_scores, sys_result, 20)
            nDCG_cal_20_sum += nDCG_cal_20
            write.writerow([ i, j, round(p_10, 3),
                             round(r_50, 3),
                             round(r_p, 3),
                             round(ap, 3),
                             round(nDCG_cal_10, 3),
                             round(nDCG_cal_20, 3) ])

        write.writerow([ i, 'mean',
                         round(p_10_sum / 10, 3),
                         round(r_50_sum / 10, 3),
                         round(r_p_sum / 10, 3),
                         round(ap_sum / 10, 3),
                         round(nDCG_cal_10_sum / 10, 3),
                         round(nDCG_cal_20_sum / 10, 3) ])


# %% TEXT ANALYSIS PART 1

def my_tokenisation(text):
    """
    :param text: string
    :return: the list of tokens
    """
    chars_to_remove = re.compile(f'[{string.punctuation}]')
    words = chars_to_remove.sub('', text).lower().split()
    return words


def remove_stop(words, stop_path):
    """
    :param words: the list of tokens
    :param stop_path: path of stopwords file
    :return: tokens after removing stopwords
    """
    stop_list = open(stop_path, "r", errors='ignore').readlines()
    stop_list = [ i.replace('\n', '') for i in stop_list ]
    n_words = list(filter(lambda x: x not in stop_list, words))
    return n_words


def my_normalisation(words):
    """
    :param words: the list of tokens
    :return: tokens after normalisation
    """
    n_words = [ stem(i) for i in words ]
    return n_words


def my_pi_index(df):
    """
    :param infos: output of my_read_xml
    :param is_prep: whether to do normalisation and remove stopwords
    :return: dict of positional inverted index
    """
    my_index = {}
    for i in df.index:
        post = df.loc[ i, 'tokens' ]
        for n, w in enumerate(post):
            if w not in my_index.keys():
                my_index[ w ] = {'fre': 0}
            else:
                pass
            my_index[ w ][ 'fre' ] += 1
            if i not in my_index[ w ].keys():
                my_index[ w ][ i ] = [ ]
            else:
                pass
            my_index[ w ][ i ].append(n)

    return my_index


def NN_cal(w, c, my_indexes, df):
    """
    :param w: the token
    :param c: the class
    :param my_indexes: my indexes
    :param df: the dataframe of samples
    :return: N-terms
    """
    U = set(my_indexes[ w ].keys())
    U.remove('fre')
    C = set(df.loc[ df[ 0 ] == c ].index)
    N = len(df)
    N11 = len(U & C) + 0.00000001
    N10 = len(U - C) + 0.00000001
    N01 = len(C - U) + 0.00000001
    N00 = N - N11 - N10 - N01
    return [ N11, N10, N01, N00 ]


def ML_cal(N, NNs):
    """calculate the ML"""
    I = NNs[ 0 ] / N * math.log((N * NNs[ 0 ]) / ((NNs[ 0 ] + NNs[ 1 ]) * (NNs[ 0 ] + NNs[ 2 ])), 2) + \
        NNs[ 1 ] / N * math.log((N * NNs[ 1 ]) / ((NNs[ 0 ] + NNs[ 1 ]) * (NNs[ 1 ] + NNs[ 3 ])), 2) + \
        NNs[ 2 ] / N * math.log((N * NNs[ 2 ]) / ((NNs[ 2 ] + NNs[ 3 ]) * (NNs[ 0 ] + NNs[ 2 ])), 2) + \
        NNs[ 3 ] / N * math.log((N * NNs[ 3 ]) / ((NNs[ 2 ] + NNs[ 3 ]) * (NNs[ 1 ] + NNs[ 3 ])), 2)

    return I


def chi_squared_cal(NNs):
    """caculate the Chi-squared"""
    I = sum(NNs) * pow((NNs[ 0 ] * NNs[ 3 ] - NNs[ 1 ] * NNs[ 2 ]), 2) / (
            (NNs[ 0 ] + NNs[ 2 ]) * (NNs[ 0 ] + NNs[ 1 ]) * (NNs[ 1 ] + NNs[ 3 ]) * (NNs[ 2 ] + NNs[ 3 ]))
    return I


def top10_printout(x):
    """get top 10 highest scores and the tokens"""
    top10 = pd.DataFrame(x).head(10).stack().to_frame()
    top10[ 'token' ], top10[ 'score' ] = top10[ 0 ].apply(lambda x: x[ 0 ]), top10[ 0 ].apply(lambda x: x[ 1 ])
    top10 = top10.reset_index()
    top10 = top10.drop(columns=[ 0 ])
    top10.columns = [ 'rank', 'class', 'token', 'score' ]
    top10.sort_values(by=[ 'class', 'rank' ], inplace=True)
    return top10


def ANALYSIS_result(my_indexes):
    """analysis the top 10 scores tokens using each methods"""
    ML_list = {'OT': [ ], 'NT': [ ], 'Quran': [ ]}
    Chi_list = {'OT': [ ], 'NT': [ ], 'Quran': [ ]}
    for i in tqdm(my_indexes.keys()):
        for c in [ 'OT', 'NT', 'Quran' ]:
            NNs1 = NN_cal(i, c, my_indexes, samples)
            ML = ML_cal(len(samples), NNs1)
            Chi_s = chi_squared_cal(NNs1)
            ML_list[ c ].append([ i, ML ])
            Chi_list[ c ].append([ i, Chi_s ])
    for k in ML_list.keys():
        ML_list[ k ].sort(key=lambda x: x[ 1 ], reverse=True)
    for k in Chi_list.keys():
        Chi_list[ k ].sort(key=lambda x: x[ 1 ], reverse=True)
    Chi_top10 = top10_printout(Chi_list)
    ML_top10 = top10_printout(ML_list)
    Chi_top10.to_csv('./output/Chi_top10.csv')
    ML_top10.to_csv('./output/ML_top10.csv')
    return Chi_top10, ML_top10


samples = pd.read_csv(r'./input/train_and_dev.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE)
tqdm.pandas(desc='preprocessing')
samples[ 'tokens' ] = samples[ 1 ].progress_apply(lambda x:
                                                  my_normalisation(remove_stop(my_tokenisation(x),
                                                                               './input/stop_words.txt')))

my_indexes = my_pi_index(samples)


Chi_top10, ML_top10 = ANALYSIS_result(my_indexes)


# %% TEXT ANALYSIS PART 2
def top10_parse(text):
    """parse the probability and ids from lda.print_topic function """
    ss = '"*+'
    chars_to_remove = re.compile(f'[{ss}]')
    words = chars_to_remove.sub(' ', text).split()
    ids = words[ 1::2 ]
    probs = words[ ::2 ]
    return ids, probs


def corpus_topic_prob(common_corpus, model):
    """get the mean topic probability of each corpus"""
    topic_scores = [ ]
    for i in tqdm(common_corpus):
        topic_scores.append(model.get_document_topics(i))

    topic_prob = pd.DataFrame()
    for n, i in tqdm(enumerate(topic_scores)):
        for t in i:
            topic_prob.loc[ n, t[ 0 ] ] = t[ 1 ]

    topic_prob[ 'class' ] = samples[ 0 ]
    class_topic = topic_prob.groupby([ 'class' ]).apply(lambda x: x.mean())
    class_topic.to_csv('./output/corpus_topics.csv')
    return class_topic


def topic_tokens(corpus_topic):
    """get the top 10 tokens of top 3 possible topics for each corpus"""
    token_result = [ ]
    for t in [ 'OT', 'NT', 'Quran' ]:
        high_1 = corpus_topic.loc[ t ].sort_values(ascending=False).head(1)
        for id in high_1.index:
            top_10 = lda.print_topic(int(id), 10)
            ids, probs = top10_parse(top_10)
            token_10 = list(map(lambda x: common_dictionary[ int(x) ], ids))
            df = pd.DataFrame({'corpus': [ t ] * 10,
                               'topic': [ id ] * 10,
                               'tokens': token_10,
                               'probs': probs}).sort_values(by='probs', ascending=False)

            df[ 'rank' ] = list(range(1, 11))
            token_result.append(df)
    topic_ana = pd.concat(token_result)
    return topic_ana


corpuses = list(samples[ 'tokens' ])
common_dictionary = Dictionary(corpuses)
common_corpus = [ common_dictionary.doc2bow(text) for text in corpuses ]
lda = LdaModel(common_corpus, num_topics=20)
corpus_topic = corpus_topic_prob(common_corpus, lda)
topic_ana = topic_tokens(corpus_topic)
topic_ana.to_csv('./output/topic_ana.csv')


# %% TEXT CLASSIFICATION

os.chdir(r'C:\Users\Simmons\PycharmProjects\ttds')

cat2id = {'Quran': 0, 'OT': 1, 'NT': 2}


def dataset_split(path1, path2, size=0.1):
    """read and split to get three dataset from the two path"""
    samples = pd.read_csv(path1, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    train_set, dev_set = train_test_split(samples, test_size=size, shuffle=True)
    test_set = pd.read_csv(path2, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    return train_set, dev_set, test_set


train_set, dev_set, test_set = dataset_split(r'./input/train_and_dev.tsv', r'./input/test.tsv')


def convert_to_bow_matrix(preprocessed_data, word2id):
    matrix_size = (len(preprocessed_data), len(word2id) + 1)
    oov_index = len(word2id)
    X = scipy.sparse.dok_matrix(matrix_size)

    for doc_id, doc in enumerate(list(preprocessed_data[ 'tokens' ])):
        for word in doc:
            X[ doc_id, word2id.get(word, oov_index) ] += 1
    return X


def data_rebuild(dataset, word2id={}):
    """
    :param dataset: the train_set
    :return: train_set after bow
    """
    if word2id == {}:
        vocabs = set(itertools.chain(*list(dataset[ 'tokens' ])))
        word2id = {}
        for word_id, word in enumerate(vocabs):
            word2id[ word ] = word_id
    X_train = convert_to_bow_matrix(dataset, word2id)
    y_train = [ cat2id[ cat ] for cat in dataset[ 0 ] ]
    return X_train, y_train, word2id


def result_report(y_real, y_pre):
    """report every metric we need"""
    report_list = [ ]
    for i in range(3):
        p1 = precision_score(y_real, y_pre, labels=[ 0 ], average=None)[ 0 ]
        r1 = recall_score(y_real, y_pre, labels=[ 0 ], average=None)[ 0 ]
        f11 = f1_score(y_real, y_pre, labels=[ 0 ], average=None)[ 0 ]
        report_list.extend([ round(p1, 3), round(r1, 3), round(f11, 3) ])

    pm = precision_score(y_real, y_pre, average='macro')
    rm = recall_score(y_real, y_pre, average='macro')
    f1m = f1_score(y_real, y_pre, average='macro')
    report_list.extend([ round(pm, 3), round(rm, 3), round(f1m, 3) ])
    return report_list


# %% baseline version
tqdm.pandas(desc='preprocessing')
train_set[ 'tokens' ] = train_set[ 1 ].progress_apply(lambda x:
                                                      my_normalisation(remove_stop(my_tokenisation(x),
                                                                                   './input/stop_words.txt')))

dev_set[ 'tokens' ] = dev_set[ 1 ].progress_apply(lambda x:
                                                  my_normalisation(remove_stop(my_tokenisation(x),
                                                                               './input/stop_words.txt')))

test_set[ 'tokens' ] = test_set[ 1 ].progress_apply(lambda x:
                                                    my_normalisation(remove_stop(my_tokenisation(x),
                                                                                 './input/stop_words.txt')))

X_train, y_train, word2id = data_rebuild(train_set)


model = svm.SVC(C=1000, kernel='rbf')
model.fit(X_train, y_train)
y_train_predictions = model.predict(X_train)
X_dev, y_dev, word2id = data_rebuild(dev_set, word2id)
y_dev_predictions = model.predict(X_dev)
X_test, y_test, word2id = data_rebuild(test_set, word2id)
y_test_predictions = model.predict(X_test)

train_rep = result_report(y_train, y_train_predictions)
dev_rep = result_report(y_dev, y_dev_predictions)
test_rep = result_report(y_test, y_test_predictions)

report_index = [ 'system', 'split', 'p-quran', 'r-quran', 'f-quran',
                 'p-ot', 'r-ot', 'f-ot',
                 'p-nt', 'r-nt', 'f-nt',
                 'p-macro', 'r-macro', 'f-macro' ]

with open(r'./output/classification.csv', 'w', newline="") as f:
    write = csv.writer(f)
    write.writerow(report_index)
    write.writerow(['baseline', 'train'] + train_rep)
    write.writerow(['baseline', 'dev'] + dev_rep)
    write.writerow(['baseline', 'test'] + test_rep)


# %% improved version
tqdm.pandas(desc='preprocessing')
# change the pre-processing method
train_set[ 'tokens' ] = train_set[ 1 ].progress_apply(lambda x: my_tokenisation(x))
dev_set[ 'tokens' ] = dev_set[ 1 ].progress_apply(lambda x: my_tokenisation(x))
test_set[ 'tokens' ] = test_set[ 1 ].progress_apply(lambda x: my_tokenisation(x))

X_train, y_train, word2id = data_rebuild(train_set)

search_result = []
# gird search for c
for c in [1, 10, 50, 100, 200]:
    model = svm.SVC(C=c, kernel='rbf')
    model.fit(X_train, y_train)
    y_train_predictions_improved = model.predict(X_train)
    X_dev, y_dev, word2id = data_rebuild(dev_set, word2id)
    y_dev_predictions_improved = model.predict(X_dev)
    dev_rep = result_report(y_dev, y_dev_predictions_improved)
    print(dev_rep)
    search_result.append(dev_rep)

# c=10 is the best
# change C
model = svm.SVC(C=10, kernel='rbf')
# model = svm.LinearSVC(c=1000)
model.fit(X_train, y_train)
y_train_predictions_improved = model.predict(X_train)
X_dev, y_dev, word2id = data_rebuild(dev_set, word2id)
y_dev_predictions_improved = model.predict(X_dev)
X_test, y_test, word2id = data_rebuild(test_set, word2id)
y_test_predictions_improved = model.predict(X_test)

train_rep = result_report(y_train, y_train_predictions_improved)
dev_rep = result_report(y_dev, y_dev_predictions_improved)
test_rep = result_report(y_test, y_test_predictions_improved)


with open(r'./output/classification.csv', 'a+', newline="") as f:
    write = csv.writer(f)
    write.writerow(['improved', 'train'] + train_rep)
    write.writerow(['improved', 'dev'] + dev_rep)
    write.writerow(['improved', 'test'] + test_rep)

