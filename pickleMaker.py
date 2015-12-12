from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
try:
   import cPickle as pickle
except:
   import pickle

clf = LogisticRegression()
dv = DictVectorizer(sparse=True)
le = LabelEncoder()
stop = stopwords.words('english')

def get_features(X) : 
    features = []
    for x in X: 
        f = {}
        words = x.split(" ")
        for i in range(len(words)):
            if words[i] not in stop:
                if words[i] in f:
                    f[words[i]] += 1
                else:
                    f[words[i]] = 1
        features.append(f)
    return features

articles = [line.strip().split('\t') for line in open('articles').readlines()] 
labels = [article[0] for article in articles]
texts = [article[1] for article in articles]
features = get_features(texts)
X = dv.fit_transform(features)
y = le.fit_transform(labels)
clf.fit(X,y)

pickle.dump(clf, open("article_clf.pkl", "wb"))
pickle.dump(dv, open("article_dv.pkl", "wb"))