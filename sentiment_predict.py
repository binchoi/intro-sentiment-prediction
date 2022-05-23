import glob
import os
import collections
import pandas as pd
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
import nltk.stem as stem
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.find('corpora/wordnet')
    nltk.find('corpora/movie_reviews')
    nltk.find('corpora/stopwords')
    nltk.find('tokenizers/punkt')
except LookupError:
    nltk.download('wordnet')
    nltk.download('movie_reviews')
    nltk.download('stopwords')
    nltk.download('punkt')

stopWords = set(stopwords.words("english"))
my_stemmer = stem.WordNetLemmatizer()

# Prepare the dataset (LOAD_REVIEWS)
def load_reviews_helper(dirname, sign):
    FileList = [(os.path.basename(fullPath), fullPath) for
                fullPath in glob.glob(dirname + f"/{sign}/*.txt")]
    FileDict = collections.defaultdict(list)
    for fileTuple in FileList:
        with open(fileTuple[1]) as f:
            text = f.read()
        FileDict['filename'].append(fileTuple[0])
        FileDict['kind'].append(sign)
        FileDict['text'].append(text)
    Df = pd.DataFrame(FileDict)
    return Df

def load_reviews(dirname):
    negDf = load_reviews_helper(dirname, "neg")
    posDf = load_reviews_helper(dirname, "pos")
    res = pd.concat([negDf, posDf], ignore_index=True, axis=0)
    return res

training_dataset = load_reviews(str(Path.home()) +
                                '/nltk_data/corpora/movie_reviews')

# Tokenization and Filtering and Lemmatization
def clean_text(text):
    return " ".join(my_stemmer.lemmatize(w) for w
                    in nltk.word_tokenize(text) if w not in stopWords)


x_training = training_dataset['text'].apply(lambda x: clean_text(x))

# Transform the long str of words (i.e. that are cleaned, tokenized,
# filtered, lemmatized) => csr_matrix (a.k.a. scipy.sparse.csr.csr_matrix)
vectorizer = TfidfVectorizer()
x_vectorized_training = vectorizer.fit_transform(x_training)

model = MultinomialNB()
model.fit(x_vectorized_training, training_dataset["kind"])

def predict_sentiment(text):
    cleaned_text = [clean_text(text)]
    textVect = vectorizer.transform(cleaned_text)
    res = model.predict(textVect)
    return res[0]
