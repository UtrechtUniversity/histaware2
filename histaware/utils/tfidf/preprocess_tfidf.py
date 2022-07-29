from utils.text_cleaner import TextCleaner
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import numpy as np

def clean_text(docs): #data_fp
    """Clean the texts stored in data_fp. Cleaning includes removing stop-words, extra spaces etc"""
    #docs = pd.read_csv(data_fp)
    txt_cleaner = TextCleaner()
    cleaned_texts = docs['p'].apply(txt_cleaner.preprocess)
    return cleaned_texts


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def train_tfidf(docs):

    # # settings that you use for count vectorizer will go here
    # tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    # # just send in all your docs here
    # tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)
    #
    # # get the first vector out (for the first document)
    # first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[1]
    # # place tf-idf values in a pandas data frame
    # df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
    #                   columns=["tfidf"])
    # df.sort_values(by=["tfidf"], ascending=False)
    # return df

    txt_cleaner = TextCleaner()
    stopwords = txt_cleaner.get_stopwords()
    cv=CountVectorizer(max_df=0.85,stop_words=stopwords, max_features=20000,ngram_range=(1, 3))
    word_count_vector=cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    return tfidf_transformer,cv


def apply_tfidf(tfidf_transformer,cv,docs_test):
    # you only needs to do this once, this is a mapping of index to
    feature_names = cv.get_feature_names()

    # get the document that we want to extract keywords from
    #doc = docs_test[0]
    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform(docs_test))

#######
    # tf_idf_vector_a = tf_idf_vector.A
    # idx_lst=[]
    # for itm in tf_idf_vector_a:
    #     idx_lst+=list(itm.argsort()[-10:])
    #
    # data = Counter(idx_lst)
    # idx_common = [it[0] for it in data.most_common(10)]
    # keywords = [feature_names[idx] for idx in idx_common ]
    # print(keywords)
    # return keywords

########


        # # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    # now print the results
    # print("\n=====Doc=====")
    # print(doc)
    print("\n===Keywords===")
    # for k in keywords:
    #     print(k, keywords[k])

    keywords_lst = list(keywords.keys())
    return keywords_lst

##############