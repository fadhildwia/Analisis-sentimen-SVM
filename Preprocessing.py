import pandas as pd
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk



def remove_punctuation(text):
    # Happy Emoticons
    emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', ':d', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
    # Sad Emoticons
    emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
    # All emoticons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)
    
    text = ' '.join([word for word in text.split() if word not in emoticons])
    # Remove Username
    text = re.sub(r'@[\w]*', ' ', text)
    # Remove Punctuation
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
    # Remove Link
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text)
    # Remove RT
    text = re.sub(r'^RT[\s]+', ' ', text)  
    # Lower case
    text = text.lower()  
    # Remove 
    text = re.sub(r'[^\w\s]+', ' ', text)
    # Remove Number
    text = re.sub(r'[0-9]+', ' ', text)
    # Remove -
    text = re.sub(r'_', ' ', text)
    # Remove 
    text = re.sub(r'\$\w*', ' ', text)
    
    return text

stopwords_indonesia = stopwords.words('indonesian')
stp = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in stp])
    return text


def stem_text(text):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

vectorizer = TfidfVectorizer()

# Preprocessing and vectorizer for testing models
def preprocess_data(data):
    
    data = remove_punctuation(data)
    
    data = remove_stopwords(data)
    
    data = stem_text(data)

    data = vectorizer.transform([data])
    
    return data