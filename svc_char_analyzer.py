from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import NuSVC

import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPClassifier

np.random.seed(500)

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

# read data

train_data = read_data('train_samples.txt') 
train_data += read_data('validation_samples.txt') 
train_labels = read_data('train_labels.txt')
train_labels += read_data('validation_labels.txt')
test_data, test_ids = read_data('test_samples.txt', read_ids = 1)

# vectorize data

vectorizer = TfidfVectorizer(ngram_range=(3,8), max_features=25000, lowercase=True, analyzer='char')
vectorizer.fit(train_data)
X_train = vectorizer.transform(train_data)
X_test = vectorizer.transform(test_data)

# train and predict 

classifier = SVC(C = 1, kernel='rbf', gamma='scale', break_ties= True, probability= False)
classifier.fit(X_train, train_labels)
predictions = classifier.predict(X_test)

solution = np.column_stack((test_ids, predictions))
np.savetxt("svc.csv", solution, delimiter = ',', fmt = '%s', header = 'id, label')