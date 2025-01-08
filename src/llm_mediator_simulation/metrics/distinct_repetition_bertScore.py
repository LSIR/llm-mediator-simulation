import nltk
from collections import Counter
from nltk.util import ngrams
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from datetime import datetime
import time
from bert_score import score as bert_score

nltk.download('punkt_tab')

class Distinct3:
    """ Compute the percentage of unique 3-grams in the text."""
    def __init__(self):
        pass

    def score(self, text: str) -> float:
        """Computes the score of the text"""
        
        tokens = nltk.word_tokenize(text)
        #The total number of 3-grams
        three_grams = list(ngrams(tokens, 3))
        #The number of unique 3-grams
        unique_three_grams = set(three_grams)
        return 100*len(unique_three_grams) / len(three_grams) if three_grams else 0

class Repetition4:
    """Compute the percentage of sentences that contain at least one repeated 4-gram.
    """
    def __init__(self):
        pass

    def score(self, text: str) -> float:
        """Computes the score of the text"""
        
        sentences = nltk.sent_tokenize(text)
        nb_repeated_sentences = 0
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            #The list of 4-grams within the current sentence
            four_grams = list(ngrams(tokens, 4))
            #if there is at least one repeated 4-gram in the sentence:
            if len(four_grams) != len(set(four_grams)):
                nb_repeated_sentences += 1
    
        return 100*nb_repeated_sentences / len(sentences) if sentences else 0

class LexicalRepetition:
    """Compute the average percentage of repeated k-grams that occur at least n times in the text.
    """

    def __init__(self, n: int = 2, kgram: int = 4):
        """
        Args:
            n (int): Number min of occurences.
            kgram (int): The length of the k-grams.
        """
        self.n = n
        self.kgram = kgram

    def score(self, text: str) -> float:
        """Computes the score of the text"""
        
        tokens = nltk.word_tokenize(text)
        grams_list = list(ngrams(tokens, self.kgram))
        gram_counts = Counter(grams_list)
        repeated_grams = [gram for gram, count in gram_counts.items() if count >= self.n]
        return 100*len(repeated_grams) / len(grams_list) if grams_list else 0




import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class BERTScore:
    """Compute the BERTScore of the text."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Args:
            model_name (str, optional): The name of the BERT model to use. Defaults to 'bert-base-uncased'.
        """
        self.model_name = model_name
        df = pd.read_csv('src/llm_mediator_simulation/metrics/data_evaluation_tweets/prochoice_prolife.csv', index_col=0)
       
        

        self.reference_data = df['text'].tolist()
        self.n = 10

    def score(self, text: str) -> float:   
        def find_closest_texts(data, target_text, n):
            
            # Initialize the TF-IDF Vectorizer
            vectorizer = TfidfVectorizer()
            
            # Fit and transform the data
            tfidf_matrix = vectorizer.fit_transform(data + [target_text])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            
            # Find the indices of the top n closest texts
            top_n_indices = np.argsort(similarities)[-n:][::-1]
            
            return [(data[i], similarities[i]) for i in top_n_indices]
            
        
        top_closest_texts = find_closest_texts(self.reference_data, text, self.n)
        closest_texts, similarities = zip(*top_closest_texts)

        text = [text]*self.n
        
        #calculate the BERTScore and take the maximum value of F1 SCORE
        P, R, F1 = bert_score(text, list(closest_texts), model_type=self.model_name, lang='en', rescale_with_baseline=True)
        return F1.max().item()
