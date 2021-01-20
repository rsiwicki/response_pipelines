# Disaster Response Pipeline Project

### Summary

This project is designed to demonstrate basic ETL (Extract, Transform, Load) and Natural Language Processing (NLP) capabilities against a data set of social media messages that are derived from people that have been subjected to natural disaster conditions. In such situations it is important for emergency response workers to understand if a message - e.g. a tweet, is a request for help for example, representing signal or general noise of less importance to the response worker. The messages can also be classified into topics that help the classification of messages to engender an appropriate response.

This example utilises a RandomForest algorithm with multi-output classifications. It also demonstrates the tokenisation approach and simple web integration using python's flask.

The key files in the project are:

apps/run.py - this starts the python flask webapp (i.e. bash> python run.py).
models/train_classifier.py - this processes cleaned data in a sqlite database and builds the machine learning model and also evaluates it. The results of the model are saved in a pkl file.
data/process_data.py - this extracts data from the categories.csv and messages.csv and transforms and loads the data into the sqlite database used by the downstream process initiated by train_classifier.py. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
