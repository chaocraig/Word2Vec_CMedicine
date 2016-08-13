# coding: utf-8



from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)



path_cmedicine_formula_components = "../cmedicine/cray_formula_components.csv"

stoplist = set('炙 炒 君'.split())


reload(sys)
sys.setdefaultencoding("utf-8")



def read_cmedicine_formula():
    with open(path_cmedicine_formula_components, 'r') as f:
       data = f.read()
    f.closed

    lines = data.split("\n")
    list_formulas, list_components = [], []
    for id, line in enumerate(lines):
        components = line.split()
        if (len(components)<=1):
            continue
        list_formulas.append(components[0])
        list_components.append(components[1:])
    return list_formulas, list_components



def build_texts(list_components):
    texts = [[component for component in components if component not in stoplist]
             for components in list_components]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    #pprint(texts)
    return texts


def build_dictionary(texts):
    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
    #print(dictionary)
    #print(dictionary.token2id)
    return dictionary


def build_corpus(dictionary, texts):
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/cemdicine.corpus', corpus) # store to disk, for later use
    #pprint(corpus)
    return corpus



def build_tfidf(list_formula, corpus):
    print "Start TfIdf..."
    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]
    for id, doc in enumerate(corpus_tfidf):
        doc_str = [(doc_vec[0], "%1.4f"%doc_vec[1]) for doc_vec in doc]
        #print u'{0}:{1}-->{2}'.format(id, list_formula[id], doc_str)
    return tfidf, corpus_tfidf



def build_Lsi(dictionary, corpus_model):
    lsi = models.LsiModel(corpus_model, id2word=dictionary, num_topics=2) # initialize an LSI transformation
    corpus_lsi = lsi[corpus_model] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    #pprint ( lsi.print_topics(2) )
    return lsi, corpus_lsi



def build_Lsi_similarity(lsi, corpus, list_formula, list_components, corpus_model):
    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
    print "LSI Similarity Start>>>>>"
    for i, corpus1 in enumerate(corpus_model):
        vec_lsi = lsi[corpus1]
        formula_similarities = index[vec_lsi]
        str_components1 = ""
        for component in list_components[i]:
            str_components1 = str_components1 + "+" + str(component)

        #print u'\n{0}:{1}:{2}'.format(i, list_formula[i], str_components1)

        for j, corpus2 in enumerate(corpus_model):
            str_components2 = ""
            for component in list_components[j]:
                str_components2 = str_components2 + "+" + str(component)
            #print u'    {0}:{1}:{2} --> {3}'.format(j, list_formula[j], str_components2, "%.4f"%formula_similarities[j])
    print "LSI Similarity End<<<<<"



def save_Lsi_similarity(lsi, corpus, list_formula, corpus_model):
    output = ""
    f_out = open("./lsi_similarities.csv", 'wb+', 0)  #zero buffer

    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
    print "Saving LSI Similarity Start>>>>>"
    len_corpus_model = len(corpus_model)
    for i, corpus1 in enumerate(corpus_model):
        vec_lsi = lsi[corpus1]
        formula_similarities = index[vec_lsi]

        for j, corpus2 in enumerate(corpus_model):
            if j < i:
                continue
            line = u'{0},{1},\"{2}\",\"{3}\",{4}\n'.format( i, j, list_formula[i], list_formula[j],
                                                            "%.4f"%formula_similarities[j] )
            output += line
            print line
    print "Saving LSI Similarity End<<<<<"

    f_out.write(output)
    f_out.close()




def build_word2vec(list_formula, list_components):
    print "Start Word2Vec..."
    #bigram_transformer = models.Phrases(list_components)
    #word2vec = models.Word2Vec(bigram_transformer[list_components], min_count=1, size=100, workers=4)
    sentences = []
    for id, components in enumerate(list_formula):
        sentence = models.doc2vec.LabeledSentence( words=list_components[id], tags=["%04d"%(id)] )
        sentences.append( sentence )

    doc2vec = models.Doc2Vec(sentences, min_count=1, size=100, workers=4)
    corpus_word2vec = []
    for id, components in enumerate(list_components):
        vec = doc2vec.infer_vector(components)
        corpus_word2vec.append( [id, vec] )
        #print u'{0}:{1}-->{2}'.format(id, list_formula[id], "%1.4f"%vec)

    return doc2vec, corpus_word2vec, sentences



def build_word2vec_similarity(list_formula, model, sentences):
    print "build_formula_similarity()..."
    for epoch in range(10):
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate`
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    for id, formula in enumerate(list_formula):
        try:
            similar_formulas = model.docvecs.most_similar("%04d"%(id))
            str_similar_formula = ""
            for formula_info in similar_formulas:
                str_similar_formula += "(%s:%s,%s)"%(formula_info[0],list_formula[int(formula_info[0])],formula_info[1])
            #print "%04d"%(id), "%s -> %s\n" % (formula, str_similar_formula)
        except:
            pass




def save_word2vec_similarity(list_formula, model, sentences):

    f_out = open("./word2vec_similarities.csv", 'wb+', 0)  #zero buffer
    output = ""

    print "build_formula_similarity()..."
    for epoch in range(10):
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate`
        model.min_alpha = model.alpha  # fix the learning rate, no decay


    for id, formula in enumerate(list_formula):
        try:
            similar_formulas = model.docvecs.most_similar("%04d"%(id))
            for formula_info in similar_formulas:
                line = "%d,%s,\"%s\",\"%s\",%s\n"%(id, formula_info[0], formula, list_formula[int(formula_info[0])], "%.4f"%formula_info[1])
                output += line
                print line
        except:
            pass

    f_out.write(output)
    f_out.close()




def get_list_xy(lsi, corpus_lsi):
    #lsi.print_topics(2)
    list_x, list_y = [], []
    for corpus in corpus_lsi:
        if len(corpus) != 2:
            continue
        list_x.append(corpus[0][1])
        list_y.append(corpus[1][1])

    return list_x, list_y



def draw_Lsi_2d_scatter(lsi, corpus_lsi):

    list_x, list_y = get_list_xy(lsi, corpus_lsi)

    fig, ax = plt.subplots()
    ax.scatter(list_x, list_y, alpha=0.5)

    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    ax.set_title('Chinese Medicine 2D Chart')

    ax.grid(True)
    fig.tight_layout()

    plt.show()


def get_word2vec_2d(corpus_word2vec, type):
    X = [corpus[1] for corpus in corpus_word2vec]
    if type == "SVD":
        X_2d = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    else:
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0 )
        X_2d = tsne.fit_transform(X)

    return X_2d



def draw_word2vec_2d_scatter(list_formula, X_2d, type):

    plt.rcParams["figure.figsize"] = 24, 12  # x, y
    fig, ax = plt.subplots()
    ax.scatter(X_2d[:, 0], X_2d[:, 1])

    for id, xy in enumerate(zip(X_2d[:, 0], X_2d[:, 1])):                                       # <--
        ax.annotate(list_formula[id], xy=xy, textcoords='data', size=5)

    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    ax.set_title('Word2Vec & %s' % type)

    ax.grid(True)
    fig.tight_layout()
    plt.show()
    return


def main():
    list_formula, list_components = read_cmedicine_formula()
    texts = build_texts(list_components)
    dictionary = build_dictionary(texts)
    corpus = build_corpus(dictionary, texts)


    # TF-IDF & LSI for document similarities
    tfidf, corpus_tfidf = build_tfidf(list_formula, corpus)
    lsi, corpus_lsi = build_Lsi(dictionary, corpus_tfidf) # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    build_Lsi_similarity(lsi, corpus, list_formula, list_components, corpus_tfidf)
    save_Lsi_similarity(lsi, corpus, list_formula, corpus_tfidf)

    # Draw 2D chart
    #draw_Lsi_2d_scatter(lsi, corpus_lsi) # map into 2D


    # Word2Vec for document similarities
    word2vec, corpus_word2vec, sentences = build_word2vec(list_formula, list_components)
    build_word2vec_similarity(list_formula, word2vec, sentences)
    save_word2vec_similarity(list_formula, word2vec, sentences)


    # Draw 2D chart
    #type = "t-SNE"
    #X_2d = get_word2vec_2d(corpus_word2vec, type)
    #draw_word2vec_2d_scatter(list_formula, X_2d, type) # map into 2D




main()





