from tqdm import tqdm
from stemming.porter2 import stem
import os
import math
from gensim.corpora import HashDictionary
from gensim.models import LdaModel
import pyLDAvis.gensim_models


# %% functions
def my_read(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    post_list = []
    post = ''
    for i in tqdm(lines):
        if i != '\n':
            post += i
        else:
            if post == '':
                pass
            else:
                post_list.append(post)
                post = ''

    return post_list


def my_tokenisation(text):
    """
    :param text: string
    :return: the list of tokens
    """
    for ch in '!"#$&()*+,-./:;<=>?@[\\]^_{|}·~‘’':
        text = text.replace(ch, " ")
    text = text.lower()
    words = text.split()
    return words


def remove_stop(words, stop_path):
    """
    :param words: the list of tokens
    :param stop_path: path of stopwords file
    :return: tokens after removing stopwords
    """
    stop_list = open(stop_path, "r", errors='ignore').readlines()
    stop_list = [ i.replace('\n', '') for i in stop_list ]

    def is_stop(words):
        for w in words:
            if w not in stop_list:
                yield w

    n_words = list(is_stop(words))
    return n_words


def my_normalisation(words):
    """
    :param words: the list of tokens
    :return: tokens after normalisation
    """
    n_words = [ stem(i) for i in words ]
    return n_words


def corpus_combine(path1, path2):
    post_list1 = my_read(path1)
    post_list2 = my_read(path2)
    post_list_p1 = {}
    post_list_p2 = {}
    for n, i in tqdm(enumerate(post_list1)):
        post_list_p1[n] = my_normalisation(remove_stop(my_tokenisation(i), './input/stop_words.txt'))

    for n, i in tqdm(enumerate(post_list2)):
        post_list_p2[ n+len(post_list1) ] = my_normalisation(remove_stop(my_tokenisation(i), './input/stop_words.txt'))

    post_list_p = {}
    post_list_p.update(post_list_p1)
    post_list_p.update(post_list_p2)
    class_dict = {}
    class_dict[1] = list(post_list_p1.keys())
    class_dict[2] = list(post_list_p2.keys())
    
    return post_list_p, class_dict


def my_pi_index(infos: dict, is_prep=True):
    """
    :param infos: output of my_read_xml
    :param is_prep: whether to do normalisation and remove stopwords
    :return: dict of positional inverted index
    """
    my_index = {}
    for id, post in infos.items():
        for n, w in enumerate(post):
            if w not in my_index.keys():
                my_index[ w ] = {'fre': 0}
            else:
                pass
            my_index[ w ][ 'fre' ] += 1
            if id not in my_index[ w ].keys():
                my_index[ w ][ id ] = [ ]
            else:
                pass
            my_index[ w ][ id ].append(n)

    return my_index


def remove_low_fre(infos):
    result = {}
    for k, v in infos.items():
        if v['fre'] > 10:
            result[k] = v
    return result


def NN_cal(w, c, N, my_indexes, class_dict):
    U = set(my_indexes[w].keys())
    U.remove('fre')
    C = set(class_dict[c])
    N11 = len(U & C) + 0.00001
    N10 = len(U - C)+ 0.00001
    N01 = len(C - U)+ 0.00001
    N00 = N - N11 - N10 - N01
    # N_dict = {0:{}, 1:{}}
    # for i in range(2):
    #     for j in range(2):
    #         if i == 1 and j == 1:
    #             N_dict[i][j] = N11
    #         elif i == 1 and j == 0:
    #             N_dict[i][j] = N10
    #         elif i == 0 and j == 1:
    #             N_dict[ i ][ j ] = N01
    #         else:
    #             N_dict[ i ][ j ] = N00

    return [N11, N10, N01, N00]


def ML_cal(N,  NNs):
    I = NNs[0] / N * math.log((N * NNs[0]) / ((NNs[0] + NNs[1]) * (NNs[0] + NNs[2])), 2) + \
        NNs[1] / N * math.log((N * NNs[1]) / ((NNs[0] + NNs[1]) * (NNs[1] + NNs[3])), 2) + \
        NNs[ 2 ] / N * math.log((N * NNs[ 2 ]) / ((NNs[ 2 ] + NNs[ 3 ]) * (NNs[ 0 ] + NNs[ 2 ])), 2) + \
        NNs[ 3 ] / N * math.log((N * NNs[ 3 ]) / ((NNs[ 2 ] + NNs[ 3 ]) * (NNs[ 1 ] + NNs[ 3 ])), 2)

    return I


def chi_squared_cal(NNs):
    I = sum(NNs) * pow((NNs[0] * NNs[3] - NNs[1] * NNs[2]), 2) / ((NNs[0] + NNs[2]) * (NNs[0] + NNs[1]) * (NNs[1] + NNs[3]) * (NNs[2] + NNs[3]))
    return I




# %% main
if __name__ == '__main__':
    os.chdir(r'C:\Users\Simmons\PycharmProjects\ttds')
    post_list_p, class_dict = corpus_combine('./input/corpora/corpus1.txt', './input/corpora/corpus2.txt')
    my_indexes = remove_low_fre(my_pi_index(post_list_p))

    ML_list = []
    Chi_list =[]
    for i in tqdm(my_indexes.keys()):
        NNs1 = NN_cal(i, 1, len(post_list_p), my_indexes, class_dict)
        ML = ML_cal(len(post_list_p), NNs1)
        Chi_s = chi_squared_cal(NNs1)
        ML_list.append([i, ML])
        Chi_list.append([i, Chi_s])

    ML_list.sort(key = lambda x: x[1], reverse=True)
    Chi_list.sort(key = lambda x: x[1], reverse=True)

    # Create a corpus from a list of texts
    corpuses = list(post_list_p.values())
    common_dictionary = HashDictionary(corpuses)
    common_corpus = [ common_dictionary.doc2bow(text) for text in corpuses ]
    # Train the model on the corpus.
    lda = LdaModel(common_corpus, num_topics=10)
    topic_scores = []
    for i in tqdm(common_corpus):
        topic_scores.append(lda.get_document_topics(i))

    corpus_scores = [0] * 10
    for i in class_dict[1]:
        for j in topic_scores[i]:
            corpus_scores[j[0]] += j[1]

    corpus_scores = [s / len(topic_scores) for s in corpus_scores]
    corpus_scores = list(zip(range(10), corpus_scores))
    corpus_scores.sort(key=lambda x: x[1], reverse=True)
    for i in corpus_scores[:3]:
        lda.print_topic(i[0], 10)


    vis = pyLDAvis.gensim_models.prepare(lda, corpuses, common_dictionary)
    pyLDAvis.save_html(vis, 'lda.html')








