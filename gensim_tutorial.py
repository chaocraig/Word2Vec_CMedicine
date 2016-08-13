# coding: utf-8

# In[1]:

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# # Quick Example

# ## First, let’s import gensim and create a small corpus of nine documents and twelve features [1]:

# In[28]:

#### https://radimrehurek.com/gensim/tutorial.html#


# In[6]:

from gensim import corpora, models, similarities

corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)],
          [(9, 1.0)],
          [(9, 1.0), (10, 1.0)],
          [(9, 1.0), (10, 1.0), (11, 1.0)],
          [(8, 1.0), (10, 1.0), (11, 1.0)]]


# Corpus is simply an object which, when iterated over, returns its documents represented as sparse vectors. If you’re not familiar with the vector space model, we’ll bridge the gap between raw strings, corpora and sparse vectors in the next tutorial on Corpora and Vector Spaces.
# 
# If you’re familiar with the vector space model, you’ll probably know that the way you parse your documents and convert them to vectors has major impact on the quality of any subsequent applications.

# In this example, the whole corpus is stored in memory, as a Python list. However, the corpus interface only dictates that a corpus must support iteration over its constituent documents. For very large corpora, it is advantageous to keep the corpus on disk, and access its documents sequentially, one at a time. All the operations and transformations are implemented in such a way that makes them independent of the size of the corpus, memory-wise.

# ## Next, let’s initialize a transformation:

# In[7]:

tfidf = models.TfidfModel(corpus)


# A transformation is used to convert documents from one vector representation into another:

# In[8]:

vec = [(0, 1), (4, 1)]
print(tfidf[vec])

#Here, we used Tf-Idf, a simple transformation which takes documents represented as bag-of-words counts and applies a weighting which discounts common terms (or, equivalently, promotes rare terms). It also scales the resulting vector to unit length (in the Euclidean norm).

#Transformations are covered in detail in the tutorial on Topics and Transformations.

#To transform the whole corpus via TfIdf and index it, in preparation for similarity queries:
# In[9]:

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)


# and to query the similarity of our query vector vec against every document in the corpus:

# In[10]:

sims = index[tfidf[vec]]
print(list(enumerate(sims)))


# In[11]:

from gensim import corpora, models, similarities

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]


# This is a tiny corpus of nine documents, each consisting of only a single sentence.
# 
# First, let’s tokenize the documents, remove common words (using a toy stoplist) as well as words that only appear once in the corpus:

# In[12]:

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint   # pretty-printer
pprint(texts)


# Your way of processing the documents will likely vary; here, I only split on whitespace to tokenize, followed by lowercasing each word. In fact, I use this particular (simplistic and inefficient) setup to mimick the experiment done in Deerwester et al.’s original LSA article [1].
# 
# The ways to process documents are so varied and application- and language-dependent that I decided to not constrain them by any interface. Instead, a document is represented by the features extracted from it, not by its “surface” string form: how you get to the features is up to you. Below I describe one common, general-purpose approach (called bag-of-words), but keep in mind that different application domains call for different features, and, as always, it’s garbage in, garbage out...
# 
# To convert documents to vectors, we’ll use a document representation called bag-of-words. In this representation, each document is represented by one vector where each vector element represents a question-answer pair, in the style of:
# 
# “How many times does the word system appear in the document? Once.”
# It is advantageous to represent the questions only by their (integer) ids. The mapping between the questions and ids is called a dictionary:

# In[13]:

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
print(dictionary)


# Here we assigned a unique integer id to all words appearing in the corpus with the gensim.corpora.dictionary.Dictionary class. This sweeps across the texts, collecting word counts and relevant statistics. In the end, we see there are twelve distinct words in the processed corpus, which means each document will be represented by twelve numbers (ie., by a 12-D vector). To see the mapping between words and their ids:

# In[14]:

print(dictionary.token2id)


# To actually convert tokenized documents to vectors:

# In[17]:

new_doc = "Human Survey interaction Human"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec) # the word "interaction" does not appear in the dictionary and is ignored


# The function doc2bow() simply counts the number of occurences of each distinct word, converts the word to its integer word id and returns the result as a sparse vector. The sparse vector [(0, 1), (1, 1)] therefore reads: in the document “Human computer interaction”, the words computer (id 0) and human (id 1) appear once; the other ten dictionary words appear (implicitly) zero times.

# In[18]:

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) # store to disk, for later use
print(corpus)


# By now it should be clear that the vector feature with id=10 stands for the question “How many times does the word graph appear in the document?” and that the answer is “zero” for the first six documents and “one” for the remaining three. As a matter of fact, we have arrived at exactly the same corpus of vectors as in the Quick Example.

# # Topics and Transformations

# ## Transformation interface

# In the previous tutorial on Corpora and Vector Spaces, we created a corpus of documents represented as a stream of vectors. To continue, let’s fire up gensim and use that corpus:

# In[21]:

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm')
print(corpus)


# In this tutorial, I will show how to transform documents from one vector representation into another. This process serves two goals:
# 
# To bring out hidden structure in the corpus, discover relationships between words and use them to describe the documents in a new and (hopefully) more semantic way.
# To make the document representation more compact. This both improves efficiency (new representation consumes less resources) and efficacy (marginal data trends are ignored, noise-reduction).

# In[22]:

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model


# We used our old corpus from tutorial 1 to initialize (train) the transformation model. Different transformations may require different initialization parameters; in case of TfIdf, the “training” consists simply of going through the supplied corpus once and computing document frequencies of all its features. Training other models, such as Latent Semantic Analysis or Latent Dirichlet Allocation, is much more involved and, consequently, takes much more time.

# Transformations always convert between two specific vector spaces. The same vector space (= the same set of feature ids) must be used for training as well as for subsequent vector transformations. Failure to use the same input feature space, such as applying a different string preprocessing, using different feature ids, or using bag-of-words input vectors where TfIdf vectors are expected, will result in feature mismatch during transformation calls and consequently in either garbage output and/or runtime exceptions.

# ## Transforming vectors

# From now on, tfidf is treated as a read-only object that can be used to convert any vector from the old representation (bag-of-words integer counts) to the new representation (TfIdf real-valued weights):

# In[26]:

doc_bow = [(0, 1), (2, 1)]
print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors


# In[27]:

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)


# In this particular case, we are transforming the same corpus that we used for training, but this is only incidental. Once the transformation model has been initialized, it can be used on any vectors (provided they come from the same vector space, of course), even if they were not used in the training corpus at all. This is achieved by a process called folding-in for LSA, by topic inference for LDA etc.

# Calling model[corpus] only creates a wrapper around the old corpus document stream – actual conversions are done on-the-fly, during document iteration. We cannot convert the entire corpus at the time of calling corpus_transformed = model[corpus], because that would mean storing the result in main memory, and that contradicts gensim’s objective of memory-indepedence. If you will be iterating over the transformed corpus_transformed multiple times, and the transformation is costly, serialize the resulting corpus to disk first and continue using that.

# Transformations can also be serialized, one on top of another, in a sort of chain:

# In[29]:

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi


# Here we transformed our Tf-Idf corpus via Latent Semantic Indexing into a latent 2-D space (2-D because we set num_topics=2). Now you’re probably wondering: what do these two latent dimensions stand for? Let’s inspect with models.LsiModel.print_topics():

# In[30]:

print "lsi.print_topics..."
print lsi.print_topics(2)


# (the topics are printed to log – see the note at the top of this page about activating logging)
# 
# It appears that according to LSI, “trees”, “graph” and “minors” are all related words (and contribute the most to the direction of the first topic), while the second topic practically concerns itself with all the other words. As expected, the first five documents are more strongly related to the second topic while the remaining four documents to the first topic:

# In[31]:

for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)


# Model persistency is achieved with the save() and load() functions:

# In[32]:

lsi.save('/tmp/model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('/tmp/model.lsi')


# In[ ]:



