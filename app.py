from flask import Flask, render_template, request
from flask.ext.heroku import Heroku

from nltk.corpus import stopwords
try:
   import cPickle as pickle
except:
   import pickle

app = Flask(__name__)
heroku = Heroku(app)
clf = pickle.load(open("article_clf.pkl","rb"))
dv = pickle.load(open("article_dv.pkl","rb"))
stop = stopwords.words('english')

# Set "homepage" to index.html
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/label', methods=['POST'])
def label():
    article = None
    if request.method == 'POST':
        article = request.form['article']
        X = dv.transform(get_features([article]))
        label = clf.predict(X)[0]
        if label:
            return render_template('success.html', label=True)
        else:
            return render_template('success.html', label=False)
    return render_template('index.html')

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

if __name__ == '__main__':
    app.debug = True
    app.run()