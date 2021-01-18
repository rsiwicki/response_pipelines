import sys

# import libraries
import re
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    engine = create_engine("sqlite:///%s" % database_filepath, execution_options={"sqlite_raw_colnames": True})
    df = pd.read_sql_table("Test1", engine)
    return df['message'],df[df.columns.difference(['message','genre','original','id'])],df[df.columns.difference(['message','genre','original','id'])].columns


def tokenize(text):
    # is to normalise case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    # tokenise the text
    tokens = word_tokenize(text)
    #lemmatize take out the stop words
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]


def build_model():
    # note can use pipeline get params to get the features to play with
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',RandomForestClassifier())
    ])
    
    parameters = {
    'features__text_pipeline_vect_ngram_range' : ((1,1),(1,2)),
    'clf__n_estimators' : [50, 100,200]
    }   

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    
    

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()