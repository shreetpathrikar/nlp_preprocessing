from flask import Flask,render_template,Response,request
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import csv
from flask import make_response
import inflect 
q = inflect.engine() 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.stem import wordnet 
from nltk import pos_tag 
from nltk import pos_tag, ne_chunk 
import numpy as np
from nltk.tag import StanfordNERTagger
from nltk.probability import FreqDist
from nltk import FreqDist

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('brown')
nltk.download('movie_reviews')

import spacy
nlp = spacy.load("en_core_web_sm")

import spacy.cli
spacy.cli.download("en_core_web_md")
import en_core_web_md
nlp1 = en_core_web_md.load()

import textblob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer