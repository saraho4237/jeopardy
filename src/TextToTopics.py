import argparse
import pickle as pickle
import pandas as pd
from pprint import pprint
import gensim
import pyLDAvis
import pyLDAvis.gensim
from preprocess_data import clean_data, make_stop_words,lemmatize_words,tokenize_words
import pickle as pickle
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

class TextToTopics():

    def __init__(self,corpus,id2word):
        self.id2word=id2word
        self.bow_corpus=corpus

    def fit(self,num_topics):
        """
        Fit LDA model.
        """
        self.lda_model = gensim.models.LdaMulticore(self.bow_corpus, num_topics=num_topics, id2word=self.id2word, passes=10,workers=2)
        return(self.lda_model)

    def print_top10_topic_keywords(self):
        pprint(self.lda_model.print_topics())


def get_data(filename):
    """Load raw data from a json file and return gensim dictionary and preprocessed corpus.

    Parameters
    ----------
    filename: The path to the json data file.

    Returns
    -------
    gensim dictionary, preprocessed bag of words corpus
    """
    df,documents=clean_data(filename)
    processed_docs = documents.apply(tokenize_words)
    #create dictionary
    id2word = gensim.corpora.Dictionary(processed_docs)
    #create corpus
    texts = processed_docs
    #Term Document Frequency
    bow_corpus = [id2word.doc2bow(text) for text in texts]
    return(id2word,bow_corpus)


if __name__ == '__main__':

    id2word,bow_corpus = get_data("JEOPARDY_QUESTIONS1.json")
    ttt_model = TextToTopics(bow_corpus,id2word)
    ttt_model.fit(8)
    ttt_model.print_top10_topic_keywords()
    # Uncomment and run this to save new model:
    # with open(args.out, 'w') as f:
    #     pickle.dump(ttt_model, f)
