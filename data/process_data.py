import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load and combine two datasets 

    INPUT
    messages_filepath: str - path to the messages data
    categories_filepath: str - path to the categories data

    OUTPUT
    df - pandas dataframe 
    '''

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])
    return df


def clean_data(df):
    '''
    Data prepossessing: transform categories dataframe, 
    convert values to number, remove duplicates

    INPUT
    df - pandas dataframe   

    OUTPUT
    df - pandas dataframe 
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use the row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to  numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string

        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
    # Replace categories column in df with new category columns
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories.reindex(df.index)], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save dataframe to sql

    INPUT
    df - pandas dataframe
    database_filename: str - path to database   

    OUTPUT
    None
    
    '''
    path = "sqlite:///"+database_filename
    engine = create_engine(path)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
