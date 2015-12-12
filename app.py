from flask import Flask, render_template, request
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.heroku import Heroku

from nltk.corpus import stopwords
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction import DictVectorizer
try:
   import cPickle as pickle
except:
   import pickle

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/articleclassification'
# db = SQLAlchemy(app)
heroku = Heroku(app)
clf = pickle.load(open("article_clf.pkl","rb"))
dv = pickle.load(open("article_dv.pkl","rb"))
stop = stopwords.words('english')

# Create our database model
# class Article(db.Model):
#     __tablename__ = "articles"
#     id = db.Column(db.Integer, primary_key=True)
#     text = db.Column(db.Text, unique=True)
#     label = db.Column(db.String(0))

#     def __init__(self, text, label):
#         self.text = text
#         self.label = label

#     def __repr__(self):
#         return '<Text %r>' % self.text

# Set "homepage" to index.html
@app.route('/')
def index(): 
    return render_template('index.html')

# Save e-mail to database and send to success page
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