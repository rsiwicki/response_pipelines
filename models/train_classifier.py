"""
This application trains a NLP model using data that is stored in
the format created by a prior pipeline application which stores
the data for disaster response tweets in sqlite.
"""

import re
import nltk
import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
import numpy
import pickle

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    """Loads data from the sqlite database created by an upstream ETL pipeline.
    
        Args:
            database_filepath (str): the file path location of the sqlite database.

        Returns:
            Disaster response messages as series, categorisation datafarme and categorisation columns series.
    """
    engine = create_engine("sqlite:///%s" % database_filepath, execution_options={"sqlite_raw_colnames": True})
    df = pd.read_sql_table("messages", engine)
    return df['message'],df.drop(['message','genre','id'], axis=1),df.drop(['message','genre','id'], axis=1).columns


def tokenize(text):
    """ Tokenises the supplied text: removes non aplha (A-Za-z) and numeric (0-9) characters
        and lower case normalises text. Then applies tokeniser and lemmatizer.

        Args: Text to tokenise.

        Returns: tokensised text

    """
    # is to normalise case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    # tokenise the text
    tokens = word_tokenize(text)
    #lemmatize take out the stop words
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]


def build_model():
    """Builds the NLP model using CountVectoriser TF-IDF transformer and currently a RandomForestClassifier
       as the estimator. In order to perform grid search a parameter grid is supplied. In addition the number
       of threads is increased to the max core count of the machine.

       Args: None.

       Returns: trained model.

    """


    # note can use pipeline get params to get the features to play with
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(),n_jobs=-1))
    ])
    
    # looks like the parameters are prefixed with the pipeline name :-)
    param_grid = { 
        'clf__estimator__n_estimators': [10,30,100],
        #'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
        #'clf__estimator__max_depth' : [4,5,6,7,8],
        #'clf__estimator__criterion' :['gini', 'entropy']
    }

    # 16 cores of the ryzen :-)
    cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
    
    return cv
    
    

def evaluate_model(model, X_test, Y_test, category_names):
    """Evalutes the performance of the model and also outputs the best
    parameters from the grid search.

    Args: 
        model: the trained NLP model.
        X_test: test data from which to gain predictions.
        Y_test: test data from which to test against the model predictions.
        category_names: the names of the labels/classifications.


    """


    y_pred = model.predict(X_test)
    print(classification_report(Y_test,y_pred))
    print(model.best_params_)

def save_model(model, model_filepath):
    """Function to save the calculated model.

    Args: 
        model: the model to save.
        model_filepath: location to save the model.
    
    """
    pickle.dump(model,open(model_filepath,"wb"))


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
