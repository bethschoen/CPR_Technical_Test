########################################
# Author: Bethany Schoen
# Date: 14/06/2024
## Functional code for technical task ##
########################################
import numpy as np
import pandas as pd
import os
# for text-preprocessing
import re
import spacy
# choose language for text processing (english)
nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words
# text transformation
import ppdeep
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# hashing comparison
from difflib import SequenceMatcher
# dimensionality reduction
from sklearn.decomposition import TruncatedSVD
# ml
from sklearn.cluster import KMeans 
from sklearn.cluster import AgglomerativeClustering
# evaluation metrics
from sklearn.metrics import silhouette_score
# plotting
from matplotlib import pyplot as plt


def read_parquet_data(filename, filepath: str = None) -> pd.DataFrame:
    """
    Import climate data
    """
    if filepath:
        complete_file_path = os.path.join(filepath, filename)
    else:
        complete_file_path = filename

    df = pd.read_parquet(complete_file_path, engine='pyarrow').reset_index().rename(columns={"index":"ID"})

    return df


def pre_process_text(file_contents, remove_numbers: bool = False, stem: bool = True, remove_stop_words: bool = True):
    """
    Pre-process text to remove any junk, common words, punctuation, and tokenise text
    Parameters:
        file_contents: str - contents within the file specified
        remove_numbers: bool default False - decide whether to remove all words containing numbers
        stem: bool default False - decide whether to performing stemming
        remove_stop_words: bool default False - decide whether to remove all stop words (in SpaCy dictionary loaded above)
    """

    if file_contents == """""":       
        # contents of file have not yet been extracted
        e = 'File contents currently unknown. You must call the various text-extracting methods before text pre-processing'
        raise Exception(e)

    ## Clean text
    # Remove unicode characters
    file_contents = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \s])|(\w+:\/\/\S+)|^rt|http.+?", "", file_contents)
    # Change all white space characters to be just spaces
    file_contents = ' '.join(file_contents.split())
    # convert text to spacy doc
    my_doc_nlp = nlp(file_contents)
    # anonymise names 
    my_doc_person = " ".join([token.text if not token.ent_type_ == 'PERSON' else 'anonperson' for token in my_doc_nlp])
    my_doc_cleaned = nlp(re.sub("anonperson anonperson", "anonperson", my_doc_person))
    if remove_stop_words:
        # remove stop words
        my_doc_cleaned = [token for token in my_doc_cleaned if not token.is_stop]
    if stem:
        # lemmatize, remove stop words and "junk"
        my_doc_cleaned = [str(token.lemma_) for token in my_doc_cleaned]        
    # remove words in list that are just whitespace
    my_doc_cleaned = [word for word in my_doc_cleaned if ''.join(str(word).split()) != ""]
    # lowercase
    my_doc_cleaned = [str(word).lower() for word in my_doc_cleaned]
    if remove_numbers:
        # remove numbers
        my_doc_cleaned = [word for word in my_doc_cleaned if not any(letter.isdigit() for letter in word)]
    # concat into one string
    preprocessed_str = " ".join(my_doc_cleaned)
    # check number of words left
    n_words_in_doc = len(my_doc_cleaned)
    
    # if nothing was left after this cleaning stage, end prematurely
    if n_words_in_doc == 0:
        e = "No words left after text pre-processing. Check document contents."
        raise Exception(e)
    else:
        return preprocessed_str
    
def pre_process_text_col_in_data(data: pd.DataFrame, col_to_pre_process: str) -> pd.DataFrame:
    """
    Use the pre-processing function on all documents in a dataset
    Parameters:
        data: pd.DataFrame - the dataset containing the documents
        col_to_pre_process: str - name of the column containing the documents
    """
    data_preprocessed = data.copy()
    data_preprocessed[f"{col_to_pre_process}_preprocessed"] = data_preprocessed.apply(lambda row: pre_process_text(row[col_to_pre_process]), axis=1)

    return data_preprocessed
    

def hash_text_data_using_ssdeep(data: pd.DataFrame, col_to_hash: str, return_matrix: bool = False) -> pd.DataFrame:
    """
    Create new column in dataframe with fuzzy hash strings
    Optional: Create matrix comparing hash string simularities using SequenceMatcher
    Parameters:
        data: pd.DataFrame - table containing pre-processed text data
        col_to_hash: str - name of column with pre-processed text
        id_col: str - unique identifier of text
        return_matrix: bool - create matrix comparing similarities in has strings
    """
    data["Hashed"] = data.apply(lambda row: ppdeep.hash(row[col_to_hash]), axis=1)

    if return_matrix:
        for row in data.itertuples():
            doc = data.id_col
            first_str = row.Hashed
            data[doc] = 0.0
            for i, second_word in enumerate(list(data.Word)):
                data.at[i, first_str] = SequenceMatcher(None, first_str, second_word).ratio()

    return data

def create_document_term_matrix(data: pd.DataFrame, preprocessed_text_col: str, ngram: int = None) -> pd.DataFrame:
    """
    Construct BoW document term matrix using pre-processed text col
    Normalise rows using the sum of words in each document
    Parameters:
        data: pd.DataFrame - dataframe with text
        preprocessed_text_col: str - column with preprocessed text
        ngram: int - (optional) length of ngrams to use
    """
    # create features
    # min_df = 0.05 => remove terms that appear in less than 5% of the documents in the corpus
    if ngram:
        bow = CountVectorizer(ngram_range=(ngram, ngram), min_df=5)
    else:
        bow = CountVectorizer(min_df=5)
    X_bow = bow.fit_transform(data[preprocessed_text_col])
    X_bow_df = pd.DataFrame(X_bow.toarray(), columns=bow.get_feature_names_out())
    # count number of words per document (after preprocessing)
    word_count_df = pd.DataFrame(X_bow_df.sum(axis=1), columns=["word_count"])
    # normalise bow
    X_bow_df_with_total = pd.concat([X_bow_df, word_count_df], axis=1)
    X_bow_df_norm = X_bow_df_with_total[list(X_bow_df.columns)].divide(X_bow_df_with_total["word_count"], axis="index")
    
    return X_bow_df_norm

def reduce_dtm_using_svd(data: pd.DataFrame, n_dims: int):
    """
    To reduce high dimensionality and support visualisation, reduce dimensions of document term matrix
    Parameters:
        data: pd.DataFrame - data to be reduced
        n_dims: int - number of dimensions to reduce data to
    """
    svd = TruncatedSVD(n_components=n_dims)
    data_reduced = pd.DataFrame(svd.fit_transform(data))

    return data_reduced

def plot_3d_data(data: pd.DataFrame, col1: str, col2: str, col3: str, color: str ='b', title: str=None):
    """

    """

    fig = plt.figure(figsize=(10, 7))

    ax1 = fig.add_subplot(projection='3d')
    ax1.scatter(data[col1], data[col2], data[col3], color=color)
    ax1.set_title(title)

    return

def cluster_documents_using_kmeans(data: pd.DataFrame, n_clusters):
    """
    useful source: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
    """
    # try five times to see how clusters change with every random set of prototypes (use different seeds)
    for seed in range(5):
        print(f"\nAttempt {seed}")
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=100,
            n_init=5,
            random_state=seed,
        ).fit(data)
        cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
        # evaluate spread across clusters - top heavy may suggest poor centering of prototypes
        print(f"Number of elements assigned to each cluster: {cluster_sizes}")
        # evaluate closeness of points in clusters and distances between groups
        score = silhouette_score(data, kmeans.predict(data))
        print(f"Silhouette Score: {score}")

    # create table of results (data points and labels)
    results = pd.concat([data, pd.Series(kmeans.labels_, name="label")], axis=1)
    
    return results

def analyse_terms_in_each_cluster(data, text_col: str, ngram: int):
    """
    
    """
    # concatenate all the document text 
    group_str = ""
    for doc in data[text_col]:
        group_str += " " + doc

    group_str_df = pd.DataFrame([[group_str]], columns=["text"])

    # count the number of words that appear in that group string (BoW)
    bow = CountVectorizer(ngram_range=(ngram, ngram))
    group_bow = bow.fit_transform(group_str_df["text"])
    group_bow_df = pd.DataFrame(group_bow.toarray(), columns=bow.get_feature_names_out())

    # transpose data so that we can order words by how frequently they appear
    group_bow_df_transposed = group_bow_df.transpose().rename(columns={0:"Count"}).sort_values(by="Count", ascending=False).head(20)
    # count the total number of words and find percentage appearance for each word (easier to compare imbalanced clusters)
    total_n_words = float(group_bow_df_transposed.Count.sum())
    group_bow_df_transposed["Percent"] = group_bow_df_transposed["Count"] / total_n_words

    return group_bow_df_transposed