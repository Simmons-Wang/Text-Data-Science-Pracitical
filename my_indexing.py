import os
import xml.dom.minidom as xmldom
import src.my_pre_processing as mp
import re

os.chdir(r'C:\Users\Simmons\PycharmProjects\ttds')


def my_read_xml(path):
    xml_file = xmldom.parse(path)
    eles = xml_file.documentElement
    texts = eles.getElementsByTagName("TEXT")
    ids = eles.getElementsByTagName('DOCNO')
    headlines = eles.getElementsByTagName('HEADLINE')
    return [ids, headlines, texts]


def my_pi_index(infos: list, is_prep=True):
    my_index = {}
    for id, head, doc in zip(infos[0], infos[1], infos[2]):
        my_id = int(id.firstChild.data)
        my_hl = head.firstChild.data
        my_txt = my_hl + '\n' + doc.firstChild.data
        if is_prep:
            my_words = mp.my_normalisation(mp.remove_stop(mp.my_tokenisation(my_txt), './input/stop_words.txt'))
        else:
            my_words = mp.my_tokenisation(my_txt)
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
    :param ser_terms: terms
    :param distance: maximum distance for the terms
    :return:
    """
    ser_terms = mp.my_normalisation(mp.my_tokenisation(terms))
    try:
        t_index = my_indexes[ser_terms[0]]
    except KeyError:
        print('terms do not exist')
        return None
    docu_list = []
    for docu in t_index.keys():
        if docu == 'fre':
            continue
        try:
            # t_pos2 = [my_indexes[term][docu] for term in ser_terms[1:]]
            t_pos2 = my_indexes[ser_terms[1]][docu]
        except KeyError:
            continue
        else:
            # print(docu)
            for pos in t_index[docu]:
                # my_judge = [pos+n+1 in pos_i for n, pos_i in enumerate(t_pos2)]
                # my_judge = [i in [pos+d for d in range(distance)] for i in t_pos2]
                my_judge = list(filter(lambda x: x in [pos+d for d in range(distance+1)], t_pos2))
                if my_judge:
                    docu_list.append(docu)
                    break
                else:
                    pass

    return docu_list


# def my_bool_search(my_indexes, s_terms: list, sign='and'):
#     t_index = my_indexes[s_terms[0]]
#     if sign == 'and':
#         for docu in t_index.keys():
#             if docu == 'fre':
#                 continue
#             try:
#                 t_pos2 = [ my_indexes[ term ][ docu ] for term in s_terms[ 1: ] ]
#             except KeyError:
#                 continue
#             print(docu)
#     elif sign == 'or':
#         docus = set(sum([list(my_indexes[term].keys()) for term in s_terms], []))
#         docus.remove('fre')
#         print(docus)
#     elif sign == 'not':
#         for docu in t_index.keys():
#             if docu == 'fre':
#                 continue
#             try:
#                 t_pos2 = [ my_indexes[ term ][ docu ] for term in s_terms[ 1: ] ]
#             except KeyError:
#                 print(docu)
#                 continue
#     else:
#         pass

def my_bool_search(my_indexes, term1: str, term2: str, sign='and'):
    t1 = mp.my_normalisation(mp.my_tokenisation(term1))
    t2 = mp.my_normalisation(mp.my_tokenisation(term2))
    try:
        if ' ' in term1:
            term1_index = my_prox_search(my_indexes, term1, distance=1)
        else:
            term1_index = my_indexes[t1[0]].keys()
    except KeyError:
        return None
    try:
        if ' ' in term2:
            term2_index = my_prox_search(my_indexes, term2, distance=1)
        else:
            term2_index = my_indexes[ t2[0] ].keys()
    except KeyError:
        term2_index = []
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


def my_bool_search_single(my_index, term):
    t1 = mp.my_normalisation(mp.my_tokenisation(term))
    try:
        if ' ' in term:
            term1_index = my_prox_search(my_index, term, distance=1)
        else:
            term1_index = my_index[t1[0]].keys()
    except KeyError:
        return None
    return list(set(term1_index))


def my_read_queries(path):
    with open(path, 'r') as f:
        queries = f.readlines()
    q_bool_list = []
    single_bool_list = []
    q_prox_list = []
    for q in queries:
        id = q.split(' ')[ 0 ]
        if q.split(' ')[1][0] == '#':
            distance = re.findall(r'\#(.*?)\(', q)[0]
            terms = re.findall(r'\((.*?)\)', q)[0]
            q_prox_list.append((id, int(distance), terms))

        else:
            sign = list(filter(lambda x: x in q, [ 'AND', 'OR', 'AND NOT' ]))
            if sign:
                term1 = re.findall(r'{0}(.*?){1}'.format(id+' ', ' '+sign[ -1 ]+' '), q)[0]
                term2 = re.findall(r'{0}(.*?){1}'.format(' '+sign[ -1 ]+' ', '\n'), q)[0]
                q_bool_list.append((id, term1, term2, sign[ -1 ].lower()))
            else:
                term = re.findall(r'{0}(.*?){1}'.format(id+' ', '\n'), q)[0]
                single_bool_list.append((id, term))
    return q_bool_list, single_bool_list, q_prox_list


# %% main
if __name__ == '__main__':
    infos = my_read_xml('./input/trec.5000.xml')
    my_indexes = my_pi_index(infos)
    my_index_print(my_indexes, './output/trec_index.txt')

    path = './input/queries.boolean.txt'
    q_bool_list, single_bool_list, q_prox_list = my_read_queries(path)
    result_list = []
    for q in q_bool_list:
        result = my_bool_search(my_indexes, q[1], q[2], q[3])
        result_list.append((q[0], result))
    for q in single_bool_list:
        result = my_bool_search_single(my_indexes, q[1])
        result_list.append((q[0], result))
    for q in q_prox_list:
        result = my_prox_search(my_indexes, q[2], q[1])
        result_list.append((q[0], result))
    result_list.sort(key=lambda x: int(x[0]))
    with open(r'./output/results.boolean.txt', 'w') as f:
        for i in result_list:
            for d in i[1]:
                f.write(i[0] + ', ' + str(d) + '\n')
