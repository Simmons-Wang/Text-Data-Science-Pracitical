import os
from stemming.porter2 import stem
import matplotlib.pyplot as plt
import numpy as np
from line_profiler import LineProfiler


# %% functions
def gettext(path):
    txt = open(path, "r", errors='ignore').read()
    return txt


def my_tokenisation(text):
    for ch in '!"#$&()*+,-./:;<=>?@[\\]^_{|}·~‘’':
        text = text.replace(ch, " ")
    text = text.lower()
    words = text.split()
    return words


def remove_stop(words, stop_path):
    stop_list = open(stop_path, "r", errors='ignore').readlines()
    stop_list = [ i.replace('\n', '') for i in stop_list ]

    def is_stop(words):
        for w in words:
            if w not in stop_list:
                yield w

    n_words = list(is_stop(words))
    return n_words


# 用map再试一下，测试时间再试一下
def my_normalisation(words):
    n_words = [ stem(i) for i in words ]
    return n_words


# def my_normalisation(words):
#     n_words = list(map(stem, words))
#     return n_words
#

def fre_count(words, sorted=True):
    counts = {}
    for word in words:
        counts[ word ] = counts.get(word, 0) + 1
    if sorted:
        items = list(counts.items())
        items.sort(key=lambda x: x[ 1 ], reverse=True)
        return items
    else:
        return counts


def check_zpif(my_count, l):
    plt.plot(list(range(l)), [ i[ 1 ] for i in my_count[ 0:l ] ])
    plt.show()


def check_benford(my_count):
    head_digit = [ str(i[ 1 ])[ 0 ] for i in my_count ]
    digit_fre = fre_count(head_digit)
    di_list = [ i[ 0 ] for i in digit_fre ]
    num_list = [ i[ 1 ] for i in digit_fre ]
    plt.plot(di_list, [ i / sum(num_list) for i in num_list ])
    plt.show()


def check_leap(my_words):
    n1 = [ ]
    n2 = [ ]
    n_terms = set()
    for n, i in enumerate(my_words):
        n1.append(n + 1)
        n_terms.add(i)
        n2.append(len(n_terms))

    def est_by_ols(n1, n2):
        x = np.log(n1)
        y = np.log(n2)
        b = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) * (x - np.mean(x)))
        a = np.mean(y) - b * np.mean(x)
        k = np.power(np.e, a)
        return k, b

    k, b = est_by_ols(n1, n2)
    est_array = np.linspace(1, len(n1), num=len(n1))
    plt.plot(est_array, k * np.power(est_array, b))
    plt.plot(n1, n2)
    plt.show()
    return k, b


def test_run(f, *x):
    lp = LineProfiler()
    lp_wrapper = lp(f)
    lp_wrapper(*x)
    lp.print_stats()

# %% main


os.chdir(r'C:\Users\Simmons\PycharmProjects\ttds')
txt = gettext('./input/pg10.txt')
# # my_words = my_normalisation(remove_stop(my_tokenisation(txt), './input/stop_words.txt'))
# my_t = my_tokenisation(txt)
# my_words = my_normalisation(my_t)
# my_count = fre_count(my_words, sorted=True)
#
# # test_run(my_normalisation, my_t)
#
# check_leap(my_words)
