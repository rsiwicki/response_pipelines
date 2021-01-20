import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse_4.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier_4.pkl")
print('model loaded')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    print('console called')
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
   
    calc1_df = df.copy()
    calc1_df['message_len'] = calc1_df['message'].str.len()
    message_len_means = calc1_df['message_len'].groupby(calc1_df['genre']).mean()
    message_len_max = calc1_df['message_len'].groupby(calc1_df['genre']).max()
    genre_names = list(message_len_means.index)
    
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=message_len_means
                )
            ],

            'layout': {
                'title': 'Mean Message Length By Genre',
                'yaxis': {
                    'title': "Mean Message Length"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=message_len_max
                )
            ],

            'layout': {
                'title': 'Max Message Length By Genre',
                'yaxis': {
                    'title': "Max Message Length"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    print('go called')
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
