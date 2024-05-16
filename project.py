import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import math 

nltk.download('punkt')
nltk.download('stopwords')

documents = {
    "doc1": "this is first documentation",
    "doc2": "this is second documentation",
    "doc3": "this is unique on and running"
}
#function of remove stop word
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_words)
#filtered_documents = {doc_id: remove_stop_words(doc_text) for doc_id, doc_text in documents.items()}
#print("Filtered Documents:") filter by stop word
#for doc_id, filtered_text in filtered_documents.items():
    #print(f"{doc_id}: {filtered_text}") #run of stopping word
    ####################################
#function of stemmer
def perform_stemmer(text):
    stemmer = PorterStemmer()
    word_tokens = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in word_tokens]
    return ' '.join(stemmed_words)
#filtered_documents = {doc_id: perform_stemmer(doc_text) for doc_id, doc_text in documents.items()}
#print("Filtered Documents:")
#for doc_id, filtered_text in filtered_documents.items():
    #print(f"{doc_id}: {filtered_text}") # run of stemming + stopping word
    ########################################
#function of indexing
def indexing(documents):
    preprocessed_documents = []
    for doc_id, text in documents.items():
        text = remove_stop_words(text)
        text = perform_stemmer(text)
        preprocessed_documents.append({
            'Document': doc_id,
            'Text': text
        })
    return pd.DataFrame(preprocessed_documents)
indexed = indexing(documents)
#print(indexed)# run of indexing
#####################################
# Function to search within indexed documents
def search(query, documents):
    apply_search = []
    for doc_id, text in documents.iterrows():
        if query.lower() in text['Text'].lower():  
            apply_search.append({
                "Document": doc_id,
                "Text": text['Text']
            })
    return pd.DataFrame(apply_search)

query = input("Enter the query to search: ")
search_results = search(query, indexed)
reterived_docement=len(search_results)
#print(search_results) #run of returned document
#print("The retrived Document is:",reterived_docement) #run of reterived document
relevant_documents = {
    0: "first document",
    1: "second document"
}
def evaluation_process(search_results, relevant_documents):
    retrieved_relevant = sum(1 for doc_id in search_results['Document'] if doc_id in relevant_documents)
    precision = retrieved_relevant / len(search_results) if len(search_results) > 0 else 0
    recall = retrieved_relevant / len(relevant_documents) if len(relevant_documents) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f_measure

precision, recall, f_measure = evaluation_process(search_results, relevant_documents)
print("Precision:", precision)
print("Recall:", recall)
print("F-measure:", f_measure)
