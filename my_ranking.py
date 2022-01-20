import os
import src.my_indexing as mi
import src.my_pre_processing as mp
import math

os.chdir(r'C:\Users\Simmons\PycharmProjects\ttds')


# %%
def my_read_queries(path):
    query_dict = {}
    with open(path, 'r') as f:
        for l in f.readlines():
            terms = l.strip().split(' ')
            k = int(terms[ 0 ])
            v = ' '.join(terms[1:])
            v = mp.my_normalisation(mp.remove_stop(mp.my_tokenisation(v), './input/stop_words.txt'))
            query_dict[ k ] = v
    return query_dict


def tf(t: str, d: int, indexing: dict):
    return len(indexing[t][d])


def idf(t: str, docs: dict, N):
    signs = len(docs[t])
    return math.log(N / signs, 10)


def tfidf_score(q: list, d: int, docs: dict, N):
    score = 0
    for t in q:
        if (t in docs.keys()) and (d in docs[t]):
            w = (1 + math.log(tf(t, d, docs), 10)) * idf(t, docs, N)
            score += w
        else:
            continue
    return score


# %% main
if __name__ == '__main__':
    infos = mi.my_read_xml('./input/trec.5000.xml')
    my_indexes = mi.my_pi_index(infos)
    my_ids = [int(i.firstChild.data) for i in infos[0]]
    N = len(my_ids)
    my_queries = my_read_queries(r'./input/queries.ranked.txt')
    with open(r'./output/results.ranked.txt', 'w') as f:
        for k, v in my_queries.items():
            rank_q = list(map(lambda x: round(tfidf_score(v, x, my_indexes, N), 3), my_ids))
            rank_q = list(zip(my_ids, rank_q))
            rank_q.sort(key=lambda x: x[ 1 ], reverse=True)
            for d in rank_q[:10]:
                f.write(str(k) + ', ' + str(d[0]) + ', ' + str(d[1]) + '\n')




