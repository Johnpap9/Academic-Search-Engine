# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:00:19 2023

@author: johnp
"""

from flask import Flask,render_template,request
from bs4 import BeautifulSoup
import requests
import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
#from gensim.summarization.bm25 import bm25

app = Flask(__name__)

#Ορισμός των αλγορίθμων ανάκτησης
ALGORITHMS = {'boolean': None,'vsm': None,'bm25': None, 'tfidf':None}

def text_processing(text):
    #Αφαίρεση ειδικών χαρακτήρων και αριθμών
    text = re.sub(r'[^a-zA-Z\s]','',text)
    
    #Tokenization
    tokens = word_tokenize(text)
    
    #Αφαίρεση stop-words
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    
    #Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    #Επιστροφή των επεξεργασμένων tokens ως κείμενο
    return ' '.join(tokens)


def web_crawler(query):
    url = "https://pubmed.ncbi.nlm.nih.gov/"
    search_url = url + f"?term={query}"
    results = []
    
    #Κατέβασμα της σελίδας με αποτελέσματα της αναζήτησης
    result = requests.get(search_url)
    doc = BeautifulSoup(result.text,"html.parser")
    
    #Εξαγωγή των συνδέσμων προς τα άρθρα
    articles = [a['href'] for a in doc.find_all('a', {'class': 'docsum-title'})]
    
    #Επεξεργασία των άρθρων
    for article in articles:
        article_url = url + article
        article_result = requests.get(article_url)
        article_doc = BeautifulSoup(article_result.text,'html.parser')
        
        #Εξαγωγή μεταδεδομένων
        title = article_doc.find('h1', {'class': 'heading-title'}).text.strip()
        authors_element = article_doc.find('span', {'class': 'authors-list-item'})
        if authors_element:
            authors = [author.text.strip() for author in authors_element]
        else:
            authors = []
        #Εξαγωγή του abstract απο την σελιδα
        abstract_element = article_doc.find('div', {'id': 'abstract'})
        abstract = abstract_element.text.strip() if abstract_element else ""
        date = article_doc.find('span', {'class': 'cit'}).text.strip()
        
        #Προεπεξεργασία των κειμένων
        processed_title = text_processing(title)
        processed_authors = [text_processing(author) for author in authors]
        processed_abstract = text_processing(abstract) if abstract else ""
        authors_string = ', '.join(processed_authors)
        
        #Προσθήκη των μεταδεδομένων στην λίστα results
        results.append({
            'Τίτλος': processed_title,
            'Συγγραφείς': authors_string,
            'Περίληψη': processed_abstract,
            'Ημερομηνία Δημοσίευσης': date,
            'Σύνδεσμος': article_url
        })
        
    return results

def search_inverted_index(results):
    inverted_index = {}
    for result_index, result in enumerate(results):
        title_terms = result['Τίτλος'].split()
        authors_terms = ', '.join(result.get('Συγγραγείς', [])).split()
        abstract_terms = result.get('Περίληψη', '').split()

        terms = title_terms + authors_terms + abstract_terms
        for term in terms:
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(result_index)
    return inverted_index

def vector_space_model(results):
    corpus = []
    for result in results:
      title = result.get('Τίτλος', '')
      authors = ', '.join(result.get('Συγγραγείς', []))
      abstract = result.get('Περίληψη', '')
      document = f'{title} {authors} {abstract}'.strip()
      corpus.append(document)
      vectorizer = TfidfVectorizer()
      vectors = vectorizer.fit_transform(corpus)
    return vectors

def calculate_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    return vectors

def rank_results_with_tfidf(query_vector, documents_vector, results):
    cosine_similarities = linear_kernel(query_vector, documents_vector).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    ranked_results = [results[index] for index in related_docs_indices]
    return ranked_results

def search_tfidf(query,documents_vector, results):
    corpus = [result['Τίτλος'] +' ' +result['Συγγραφείς'] +' '+ result.get('Περίληψη',' ') for result in results]
    query_vector = ALGORITHMS['tfidf'].transform([text_processing(query)]).T
    documents_vector = calculate_tfidf(corpus)
    ranked_results = rank_results_with_tfidf(query_vector, documents_vector, results)
    return ranked_results
def bm25_model(results):
    corpus = []
    for result in results:
      title = result.get('Τίτλος', '')
      authors = ', '.join(result.get('Συγγραγείς', []))
      abstract = result.get('Περίληψη', '')
      document = f'{title} {authors} {abstract}'.strip()
      corpus.append(document)
      tokenized_corpus = [text_processing(doc).split() for doc in corpus]
      #Δημιουργία του μοντέλου BM25
      #bm25_model = bm25.BM25(tokenized_corpus)
    return bm25_model

def process_query(query):
    #Χωρίζει το query σε λέξεις
    terms = word_tokenize(query)
    
    #Προεπεξεργασία κάθε λέξης(αφαίρεση stopwords, stemming)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    processed_terms = [stemmer.stem(term.lower()) for term in terms if term.lower not in stop_words]
    
    return processed_terms

def apply_boolean_query(results, query_terms):
    #Εκτέλεση αναζήτησης boolean με βάση τις επεξεργασμένες λέξεις του query
    result_indices = set(range(len(results)))
    current_operation = None
    
    for term in query_terms:
        if term.lower() == 'and':
            current_operation = 'and'
        elif term.lower == 'or':
            current_operation = 'or'
        elif term.lower == 'not':
            current_operation = 'not'
        else:
            term_results = ALGORITHMS['boolean'].get(term, set())
            
            if current_operation == 'and':
                result_indices.intersection_update(term_results)
            elif current_operation == 'or':
                result_indices.update(term_results)
            elif current_operation == 'not':
                result_indices.difference_update(term_results)
            else:
                result_indices.update(term_results)
    return [results[index] for index in result_indices]
                
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search',methods=['POST'])
def search():
    query = request.form['query']
    author = request.form['author']
    date = request.form['date']
    #Εκτέλεση του web crawler για την αναζήτηση
    results = web_crawler(query)
    
    #Οι παρακάτων γραμμές κατασκευάζουν τα δεδομένα που απαιτούνται απο τους αλγορίθμους ανάκτησης
    global ALGORITHMS
    inverted_index = search_inverted_index(results)
    vectors = vector_space_model(results)
    bm25 = bm25_model(results)
    
    #Ενημέρωση των αλγορίθμων ανάκτησης
    ALGORITHMS['boolean'] = inverted_index
    ALGORITHMS['vsm'] = vectors
    ALGORITHMS['bm25'] = bm25
    #ALGORITHMS['tfidf'] = tfidf
    
    #Επεξεργασία του query
    query_terms = process_query(query)
    
    # Εφαρμογή Φίλτρων
    boolean_results = apply_filters(results, author, date)
    vsm_results = apply_filters(results, author, date)
    bm25_results = apply_filters(results, author, date)
    
    #Εφαρμογή Boolean Αναζήτησης
    boolean_results = apply_boolean_query(results, query_terms)
    
    #Εφαρμογή tfidf Αναζήτησης
    #tfidf_results = search_tfidf(query,documents_vector results)
    
    return render_template('results.html',query=query, results=results,boolean_results=boolean_results, vsm_results=vsm_results, bm25_results=bm25_results,)

def apply_filters(results, author, date):
    filtered_results = results.copy()
    
    #Εφαρμογή φίλτρου συγγραφέα
    if author:
        filtered_results = [result for result in filtered_results if author.lower() in str(result['Συγγραφείς']).lower()]
    
    #Εφαρμογή φίλτρου ημερομηνίας δημοσίευσης
    if date:
        filtered_results = [result for result in filtered_results if date in result['Ημερομηνία Δημοσίευσης']]
        
    return filtered_results
def search_boolean(query,results):
    terms = text_processing(query).split()
    result_indices = set()
    for term in terms:
       result_indices.update(ALGORITHMS['boolean'].get(term, set()))
    return [results[result_index] for result_index in result_indices]

def search_vsm(query,results):
    global ALGORITHMS
    query_vector = ALGORITHMS['vsm'].dot(ALGORITHMS['vsm'].transform([text_processing(query)]).T)
    cosine_similarities = linear_kernel(query_vector,ALGORITHMS,['vsm']).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    return [results[index] for index in related_docs_indices]

def search_bm25(query,results):
    global ALGORITHMS
    tokenized_query = text_processing(query).split()
    scores = ALGORITHMS['bm25'].get_scores(tokenized_query)
    related_docs_indices = [index for index, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return [results[index] for index in related_docs_indices]



def save_to_json(data, filename='web_crawler_data.json'):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data,json_file, ensure_ascii=False,indent=4)
        
if __name__ == '__main__':
    app.run(debug=True)
        

