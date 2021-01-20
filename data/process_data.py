"""ETL pipeline for extracting, transforming / cleaning and loading messages.

   The pipeline outputs a named sqlite database for down stream processing.
"""


import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the data from supplied csv files and performs some pre-prep on the unmerged data
    files

    Args:
        messages_filepath: location of the messasges .csv file.
        categories_filepath: location of the categories .csv file.

    Returns:
        DataFrame: extracted and prepared data. 

    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, left_on='id',right_on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda r : r.str.split('-')[0][0])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda c : c.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # non binary dirty related fields cause model to fail during training
    #categories['related'] = categories['related'].replace(2,1)
    # drop categories they skew model data
    categories = categories.drop(columns=['related'])

    # drop the original column
    df = df.drop('original', axis=1)
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1,join='inner')
    print(df.columns)
    return df
    
def clean_data(df):
    """ Template method for additional cleanind data steps.

    Args: 
        df: dataframe to clean.

    Returns:
        DataFrame: cleaned dataframe.

    """
    # drop duplicates
    df= df.drop_duplicates(subset=["message"],keep="first")
    df = df.dropna()
    print(df.columns)
    print(df.count)
    return df

def save_data(df, database_filename):
    """Saves / loads the processed data to a sqlite database

    Args:
        df: dataframe to save.
        database_filename: location to save the sqlite.

    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
