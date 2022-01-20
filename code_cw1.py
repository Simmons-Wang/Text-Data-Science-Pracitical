from stemming.porter2 import stem
import os
import xml.dom.minidom as xmldom
import re
import math

os.chdir(r'C:\Users\Simmons\PycharmProjects\ttds')

# %% definition of functions
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


def my_read_xml(path):
    """
    :param path: path of xml file
    :return: [id of every documents, headline of every documents, text of every documents]
    """
    xml_file = xmldom.parse(path)
    eles = xml_file.documentElement
    texts = eles.getElementsByTagName("TEXT")
    ids = eles.getElementsByTagName('DOCNO')
    headlines = eles.getElementsByTagName('HEADLINE')
    return [ ids, headlines, texts ]


def my_pi_index(infos: list, is_prep=True):
    """
    :param infos: output of my_read_xml
    :param is_prep: whether to do normalisation and remove stopwords
    :return: dict of positional inverted index
    """
    my_index = {}
    for id, head, doc in zip(infos[ 0 ], infos[ 1 ], infos[ 2 ]):
        my_id = int(id.firstChild.data)
        my_hl = head.firstChild.data
        my_txt = my_hl + '\n' + doc.firstChild.data
        if is_prep:
            my_words = my_normalisation(remove_stop(my_tokenisation(my_txt), './input/stop_words.txt'))
        else:
            my_words = my_tokenisation(my_txt)
        for n, w in enumerate(my_words):
            if w not in my_index.keys():
                my_index[ w ] = {'fre': 0}
            else:
                pass
            my_index[ w ][ 'fre' ] += 1
            if my_id not in my_index[ w ].keys():
                my_index[ w ][ my_id ] = [ ]
            else:
                pass
            my_index[ w ][ my_id ].append(n)

    return my_index


def my_index_print(ind: dict, path):
    """
    :param ind: dict of positional inverted index
    :param path: output path
    :return:
    """
    with open(path, 'w') as f:
        for k1, v1 in ind.items():
            f.write(str(k1) + ': ' + str(v1[ 'fre' ]) + '\n')
            for k2, v2 in v1.items():
                if k2 == 'fre':
                    continue
                else:
                    f.write('\t' + str(k2) + ': ' + str(v2).replace('[', '').replace(']', '') + '\n')


def my_prox_search(my_indexes, terms: list, distance: int):
    """
    :param my_indexes: my positional inverted index
    :param terms: terms
    :param distance: maximum distance for the terms
    :return: list of documents' ID we get
    """
    ser_terms = my_normalisation(my_tokenisation(terms))
    try:
        t_index = my_indexes[ ser_terms[ 0 ] ]
    except KeyError:
        print('terms do not exist')
        return None
    docu_list = [ ]
    for docu in t_index.keys():
        if docu == 'fre':
            continue
        try:
            # t_pos2 = [my_indexes[term][docu] for term in ser_terms[1:]]
            t_pos2 = my_indexes[ ser_terms[ 1 ] ][ docu ]
        except KeyError:
            continue
        else:
            # print(docu)
            for pos in t_index[ docu ]:
                # my_judge = [pos+n+1 in pos_i for n, pos_i in enumerate(t_pos2)]
                # my_judge = [i in [pos+d for d in range(distance)] for i in t_pos2]
                my_judge = list(filter(lambda x: x in [ pos + d for d in range(distance + 1) ], t_pos2))
                if my_judge:
                    docu_list.append(docu)
                    break
                else:
                    pass

    return docu_list


def my_bool_search(my_indexes, term1: str, term2: str, sign='and'):
    """
    :param my_indexes: my positional inverted index
    :param term1: A in "A operator B"
    :param term2: B in "A operator B"
    :param sign: operator in "A operator B"
    :return: list of documents' ID we get
    """
    t1 = my_normalisation(my_tokenisation(term1))
    t2 = my_normalisation(my_tokenisation(term2))
    try:
        if ' ' in term1:
            term1_index = my_prox_search(my_indexes, term1, distance=1)
        else:
            term1_index = my_indexes[ t1[ 0 ] ].keys()
    except KeyError:
        return None
    try:
        if ' ' in term2:
            term2_index = my_prox_search(my_indexes, term2, distance=1)
        else:
            term2_index = my_indexes[ t2[ 0 ] ].keys()
    except KeyError:
        term2_index = [ ]
    if sign == 'and':
        # docus = [ set(my_indexes[ term ].keys()) for term in s_terms ]
        # docus = set(docus[ 0 ]).intersection(*docus[ 1: ])
        docus = set(term1_index).intersection(set(term2_index))
        if 'fre' in docus:
            docus.remove('fre')
    elif sign == 'or':
        # docus = set(sum([list(my_indexes[term].keys()) for term in s_terms], []))
        docus = set(term1_index).union(set(term2_index))
        if 'fre' in docus:
            docus.remove('fre')
    elif sign == 'and not':
        docus = set(term1_index).difference(set(term2_index))
    else:
        print("your sign is wrong")
    return list(docus)


def my_phrase_search(my_index, term):
    """
    :param my_index: my positional inverted index
    :param term: term we want to search for
    :return: list of documents' ID we get
    """
    t1 = my_normalisation(my_tokenisation(term))
    try:
        if ' ' in term:
            term1_index = my_prox_search(my_index, term, distance=1)
        else:
            term1_index = my_index[ t1[ 0 ] ].keys()
    except KeyError:
        return None
    return list(set(term1_index))


def my_read_queries_s(path):
    """
    :param path: path of queries.boolean.txt
    :return: list of queries for Boolean search, list of queries for Phrase search, list of queries for Proximity search
    """
    with open(path, 'r') as f:
        queries = f.readlines()
    q_bool_list = [ ]
    single_bool_list = [ ]
    q_prox_list = [ ]
    for q in queries:
        id = q.split(' ')[ 0 ]
        if q.split(' ')[ 1 ][ 0 ] == '#':
            distance = re.findall(r'\#(.*?)\(', q)[ 0 ]
            terms = re.findall(r'\((.*?)\)', q)[ 0 ]
            q_prox_list.append((id, int(distance), terms))

        else:
            sign = list(filter(lambda x: x in q, [ 'AND', 'OR', 'AND NOT' ]))
            if sign:
                term1 = re.findall(r'{0}(.*?){1}'.format(id + ' ', ' ' + sign[ -1 ] + ' '), q)[ 0 ]
                term2 = re.findall(r'{0}(.*?){1}'.format(' ' + sign[ -1 ] + ' ', '\n'), q)[ 0 ]
                q_bool_list.append((id, term1, term2, sign[ -1 ].lower()))
            else:
                term = re.findall(r'{0}(.*?){1}'.format(id + ' ', '\n'), q)[ 0 ]
                single_bool_list.append((id, term))
    return q_bool_list, single_bool_list, q_prox_list


def my_search_result_print(q_bool_list, single_bool_list, q_prox_list, my_indexes):
    """
    :param q_bool_list: list of queries for Boolean search
    :param single_bool_list: list of queries for Phrase search
    :param q_prox_list: list of queries for Proximity search
    :param my_indexes: my positional inverted index
    :return:
    """
    result_list = [ ]
    for q in q_bool_list:
        result = my_bool_search(my_indexes, q[ 1 ], q[ 2 ], q[ 3 ])
        result_list.append((q[ 0 ], result))
    for q in single_bool_list:
        result = my_phrase_search(my_indexes, q[ 1 ])
        result_list.append((q[ 0 ], result))
    for q in q_prox_list:
        result = my_prox_search(my_indexes, q[ 2 ], q[ 1 ])
        result_list.append((q[ 0 ], result))
    result_list.sort(key=lambda x: int(x[ 0 ]))
    with open(r'./output/results.boolean.txt', 'w') as f:
        for i in result_list:
            for d in i[ 1 ]:
                f.write(i[ 0 ] + ', ' + str(d) + '\n')


def my_read_queries_r(path):
    """
    :param path: path of queries.ranked.txt
    :return: dicts of queries, format: {id: list of tokens}
    """
    query_dict = {}
    with open(path, 'r') as f:
        for l in f.readlines():
            terms = l.strip().split(' ')
            k = int(terms[ 0 ])
            v = ' '.join(terms[ 1: ])
            v = my_normalisation(remove_stop(my_tokenisation(v), './input/stop_words.txt'))
            query_dict[ k ] = v
    return query_dict


def tf(t: str, d: int, indexing: dict):
    """
    :param t: token
    :param d: document
    :param indexing: my positional inverted index
    :return: tf
    """
    return len(indexing[ t ][ d ])


def idf(t: str, docs: dict, N):
    """
    :param t: token
    :param docs: my positional inverted index
    :param N: numbers of document
    :return: idf
    """
    signs = len(docs[ t ])
    return math.log(N / signs, 10)


def tfidf_score(q: list, d: int, docs: dict, N):
    """
    :param q: tokens of a term
    :param d: document
    :param docs: my positional inverted index
    :param N: numbers of document
    :return: tfidf_score
    """
    score = 0
    for t in q:
        if (t in docs.keys()) and (d in docs[ t ]):
            w = (1 + math.log(tf(t, d, docs), 10)) * idf(t, docs, N)
            score += w
        else:
            continue
    return score


def my_ranking_print(query, my_indexes, my_ids):
    """
    :param query: queries
    :param my_indexes: my positional inverted index
    :param my_ids: IDs of doucuments
    :return:
    """
    N = len(my_ids)
    with open(r'./output/results.ranked.txt', 'w') as f:
        for k, v in query.items():
            rank_q = list(map(lambda x: round(tfidf_score(v, x, my_indexes, N), 3), my_ids))
            rank_q = list(zip(my_ids, rank_q))
            rank_q.sort(key=lambda x: x[ 1 ], reverse=True)
            for d in rank_q[ :10 ]:
                f.write(str(k) + ', ' + str(d[ 0 ]) + ', ' + str(d[ 1 ]) + '\n')


# %% main
if __name__ == '__main__':
    infos = my_read_xml('./input/trec.5000.xml')  # input the collections
    my_indexes = my_pi_index(infos)  # positional inverted index
    my_index_print(my_indexes, './output/trec_index.txt')  # output the positional inverted indexes
    path = './input/queries.boolean.txt'
    q_bool_list, single_bool_list, q_prox_list = my_read_queries_s(path)  # input the queries.boolean
    my_search_result_print(q_bool_list, single_bool_list, q_prox_list, my_indexes)  # output the result of boolean search

    my_ids = [ int(i.firstChild.data) for i in infos[ 0 ] ]

    my_queries = my_read_queries_r(r'./input/queries.ranked.txt') # input the queries.ranked
    my_ranking_print(my_queries, my_indexes, my_ids)  # output the result of ranking
