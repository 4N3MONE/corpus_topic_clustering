from sklearn.feature_extraction.text import CountVectorizer
from mecab import MeCab
from bertopic import BERTopic
import json
import sys
import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    preprocessed_documents = construct_data(sys.argv[1])
    custom_tokenizer = CustomTokenizer(Kiwi())
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)
    model = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
    		vectorizer_model=vectorizer,
            nr_topics=3, # 문서를 대표하는 토픽의 갯수
            top_n_words=10)
    print('start fitting...')
    topics, probs = model.fit_transform(preprocessed_documents)
    model.get_topic_info()
    model.visualize_topics()
    model.visualize_distribution(probs[0])
