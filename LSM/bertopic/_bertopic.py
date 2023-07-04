from sklearn.feature_extraction.text import CountVectorizer
from mecab import MeCab
from bertopic import BERTopic
import json
import sys
import tqdm

class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        word_tokens = self.tagger.nouns(sent)
        result = [word for word in word_tokens if len(word) > 1]
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
    preprocessed_documents = construct_data(sys.argv[1])
    custom_tokenizer = CustomTokenizer(MeCab())
    
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)
    model = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens", \
                 vectorizer_model=vectorizer,
                 nr_topics=50,
                 top_n_words=10,
                 calculate_probabilities=True)
    print('start fitting...')
    topics, probs = model.fit_transform(preprocessed_documents)
    
    model.visualize_topics()
    model.visualize_distribution(probs[0])