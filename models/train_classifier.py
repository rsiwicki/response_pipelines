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




nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    engine = create_engine("sqlite:///%s" % database_filepath, execution_options={"sqlite_raw_colnames": True})
    df = pd.read_sql_table("messages", engine)
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
    
    # looks like the parameters are prefixed with the pipeline name :-)
    param_grid = { 
        #'clf__n_estimators': [200, 500],
        #'clf__max_features': ['auto', 'sqrt', 'log2'],
        #'clf__max_depth' : [4,5,6,7,8],
        'clf__criterion' :['gini', 'entropy']
    }

    cv = GridSearchCV(pipeline, param_grid=param_grid)
    
    return cv
    
    

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    labels = numpy.unique(y_pred)
    print("Confusion Matrix:\n", confusion_matrix(Y_test,y_pred,labels=labels))
    print("Accuracy = ",metrics.accuracy_score(Y_test, y_pred)*100)
    print("F1 = ", metrics.f1_score(Y_test, y_pred)*100)
    print("Recall = ", metrics.recall_score(Y_test, y_pred)*100)
    print("Precision = ", metrics.precision_score(Y_test, y_pred)*100)
    print("Best Params:\n", model.best_params_)


def save_model(model, model_filepath):
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
        model.fit(X_train, Y_train.values.ravel())
        
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