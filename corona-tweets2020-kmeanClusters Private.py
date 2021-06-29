import spacy
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans

nlp = spacy.load('en_core_web_lg')
f = pd.read_csv('txeet.csv', dtype='string')

txraw = []
for i in range(len(f)):
    txraw.append(f.loc[i].to_list())

trc = txraw.copy()

notalpha = []
for i in txraw:
    for j in i[3]:
        if j.isalpha() == False and j.isdigit() == False and j not in notalpha:
            notalpha.append(j)

notalpha.remove("'")  # "'" and "," are needed, removing them
notalpha.remove(",")  # from the blacklist.
notalpha.insert(0, "&amp")  # An annoying character

txrawcopy = []
for i in range(len(trc)):
    x = trc[i][3]
    X = x.split()
    tmp = []
    for k in range(len(X)):
        if 'http' not in X[k] and '@' not in X[k] and '#' not in X[k]:
            for nal in notalpha:
                if nal in X[k]:
                    X[k] = X[k].replace(f'{nal}', ' ')

            X[k] = X[k].replace('wanna', 'want to')
            X[k] = X[k].replace('gonna', 'going to')
            X[k] = X[k].replace('  ', ' ')
            for j in ['corona', 'virus',
                      'covid']:  # spacy does not recognize new names given to covid-19. replacing all possibilities with "virus".
                if j in X[k].lower():
                    X[k] = 'virus'

            tmp.append(X[k])

        tmp2 = ' '.join(tmp)
    if tmp2.__len__() > 0:
        while '  ' in tmp2:
            tmp2 = tmp2.replace('  ', ' ')

    tmp2 = tmp2.strip()
    txrawcopy.append([trc[i][0], trc[i][1], trc[i][2], tmp2])

twtchunks2 = []
for i in txrawcopy:
    cc = nlp(i[3]).noun_chunks
    uu = []
    tm = []
    for k in cc:
        uu.append(k.lemma_)
        nn = []
        for j in k:
            jp = j.pos_
            jl = j.lemma_
            if jp != 'PRON' and jp != 'CCONJ' and jp != 'ADV' and jp != 'DET' and \
                    len(jl) > 2 and j.is_oov == False and 'd' not in j.shape_:
                nn.append(j.lemma_.lower())
        Y = ' '.join(nn)
        if Y.__len__() > 0:
            tm.append(Y)
    if tm.__len__() > 0:
        twtchunks2.append([i, uu, tm])

voc = []
for i in twtchunks2:
    for j in i[2]:
        voc.append(j)

noun_count = pd.Series(voc).value_counts().reset_index()
noun_count.rename(columns={'index': 'Nouns', 0: 'No. of usage'}, inplace=True)
noun_count.to_csv('noun count.csv', index=False)

top20 = noun_count.loc[:19]['Nouns'].to_list()

# Above lines are the same as Corona-tweet2020
# Below is the new lines

# Change twtsample value for smaller or bigger sample.
# 10 categories should be enough. Chnage n_clusters for different no. of categories. Set verbosity to 1 if needed.

twtsample= 1000
categories=10
verbosity=0
noun_vectors = []
twt_sel = random.choices(twtchunks2, k=twtsample)
for i in twt_sel:
    noun_vectors.append(nlp(' '.join(i[2])).vector)

km = KMeans(n_clusters=categories, init='k-means++', verbose=verbosity)
labels = km.fit_predict(noun_vectors)


def tweetClassififcation(selection=0):
    """ Returns the list and the no. of tweets associated with each label."""

    # Gather the number of tweets for each label
    label_length = []
    for i in np.unique(labels):
        label_length.append([i, len(np.array(twt_sel, dtype=object)[labels == i])])
    # Read the tweets of each selection
    collection = []
    for i in np.array(twt_sel, dtype=object)[labels == selection][:, 0]:
        collection.append(i[3])

    return label_length, collection
