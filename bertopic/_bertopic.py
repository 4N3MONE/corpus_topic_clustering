import json
import pickle
import glob, os
import pandas as pd
from kiwipiepy import Kiwi
from bertopic import BERTopic
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import sys
import argparse
import pyLDAvis
import numpy as np
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Hello MY Name is Seungmin! Nice to meet you~")
parser.add_argument('-p', '--path', type=str, help='Enter your relative json path')
parser.add_argument('-e', '--embedding_model', type=str, default='sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
parser.add_argument('-n', '--nr_topics', type=int, default=3, help="Enter nr_topics the number of topics representing document")
parser.add_argument('-t', '--top_n_words', type=int, default=10, help="Enter top_n_words")

# 불용어를 정의한다
user_stop_word = ["안녕", "안녕하세요", "때문", "지금", "감사", "네", "감사합니다"]

# 토크나이저에 명사만 추가한다
extract_pos_list = ["NNG", "NNP"]
class CustomTokenizer:
  def __init__(self, kiwi):
    self.kiwi = kiwi
  def __call__(self, text):
    result = list()
    for word in self.kiwi.tokenize(text):
# 명사이고, 길이가 2이상인 단어이고, 불용어 리스트에 없으면 추가하기
      if word[1] in extract_pos_list and len(word[0]) > 1 and word[0] not in user_stop_word:
        result.append(word[0])
    return result


def construct_data(json_file_path: str):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    for d in data['data']:
        documents.append(d['text'])

    preprocessed_documents = []

    for line in documents:
        if line and not line.replace(' ', '').isdecimal():
            preprocessed_documents.append(line)
    
    return preprocessed_documents
            
if __name__=="__main__":
    args = parser.parse_args()
    
    preprocessed_documents = construct_data(args.path)
    custom_tokenizer = CustomTokenizer(Kiwi())
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)
    model = BERTopic(embedding_model=args.embedding_model,
    		vectorizer_model=vectorizer,
            nr_topics=args.nr_topics, # 문서를 대표하는 토픽의 갯수
            top_n_words=args.top_n_words,
            calculate_probabilities=True)
   
    
    print('start fitting...')
    topics, probs = model.fit_transform(preprocessed_documents)
    df = model.get_topic_info()
    print(df)
    fig = model.visualize_topics()
    fig.write_html("/home/yys/corpus_topic_clustering/LSM/bertopic/fig1.html")
    fig2 = model.visualize_distribution(probs[0])
    fig2.write_html("/home/yys/corpus_topic_clustering/LSM/bertopic/fig2.html")
