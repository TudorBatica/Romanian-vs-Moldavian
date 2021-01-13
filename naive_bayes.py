#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
#from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer

def read_data(file_path, read_ids = 0):
  with open(file_path, 'r') as file:
    file_data = file.readlines()
  data = []
  ids = []
  for sample in file_data:
    id, content = sample.split('\t') 
    content = content.rstrip()
    data.append(content)
    if(read_ids == 1):
      ids.append(id)
  if(read_ids == 0):
    return data
  return data, ids

train_data = read_data('train_samples.txt') 
train_labels = read_data('train_labels.txt')

#test_data = read_data('validation_samples.txt')
#test_labels = read_data('validation_labels.txt')
test_data, test_ids = read_data('test_samples.txt', read_ids = 1)

#vectorize data

vectorizer = TfidfVectorizer(ngram_range = (1, 7), max_features = 35000)
vectorizer.fit(train_data)
X_train = vectorizer.transform(train_data)
X_test = vectorizer.transform(test_data)

# train and predict

classifier = MultinomialNB()
classifier.fit(X_train, train_labels)
predictions = classifier.predict(X_test)

#print('F1-score is {:.4%}'.format(f1_score(test_labels, predictions, pos_label='1')))

solution = np.column_stack((test_ids, predictions))
np.savetxt("naivebayes2.csv", solution, delimiter = ',', fmt = '%s', header = 'id, label')