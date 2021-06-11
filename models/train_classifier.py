import sys
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    Load and transform data for classification 

    INPUT
    database_filepath: str - path to database   

    OUTPUT
    X: pandas dataframe - features for classification
    Y: pandas dataframe - labels for classification 36 categories
    category_names: list - all categories names
    
    '''
    # load to database
    path = "sqlite:///"+database_filepath
    engine = create_engine(path)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    
    # define features and label array
    X = df["message"]
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # define labels names
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    '''
    Prepossessing and cleaning data 

    INPUT
    text: str - message   

    OUTPUT
    lemmatized: array - prepossessed and cleaned text 
    
    '''
    # remove punctuation from string except apostrophe e.g. don’t  
    custom_punctuation = punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', custom_punctuation))

    # tokenize text
    tokens = word_tokenize(text)

    # create list with lemmas and lowercase all tokens
    lemmatized = [WordNetLemmatizer().lemmatize(w.lower().rstrip())
                  for w in tokens]
    return lemmatized


def build_model(grid_search):
    '''
    Build/tune model

    INPUT
    grid_search: bool - if True do parameter tuning    

    OUTPUT
    pipeline: sklearn - tuned model for MultiOutput Classification 
    
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier())),
    ])

    #print(pipeline.get_params())
    if grid_search == "True":
        print("Start Model Tuning")
        clf__max_depth = [int(x) for x in np.linspace(start = 5, stop = 15, num = 3)]
        parameters = {'clf__estimator__metric': ["minkowski", "manhattan", "euclidean", "chebyshev"],
                    'clf__estimator__n_neighbors': clf__max_depth,
                  }
        cv = GridSearchCV(pipeline,  param_grid=parameters)
        return  cv
    else:
        best_params = joblib.load("best_params.pkl")
        pipeline.set_params(**best_params)
        return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model:  accuracy, precision, recall  f1-score

    INPUT
    model: sklearn model - trained MultiOutput Classification
    X_test: numpy.ndarray - features 
    Y_test: numpy.ndarray - labels
    category_names: list - labels/ categories namen   

    OUTPUT
    None
    
    '''
    # prediction on test data
    y_pred = model.predict(X_test)

    # calculate accuracy
    accuracy = (y_pred == Y_test).mean()

    print("Labels:", category_names)
    print("Accuracy:", accuracy)

    # calculate precision, recall  f1-score for each categorie
    for i, column in enumerate(y_pred.T):
        predicted = list(column)
        y_column = list(Y_test[category_names[i]].values)
        print(category_names[i])
        print(classification_report(y_column, predicted))


def save_model(model, model_filepath):
    '''
    save trained model

    INPUT
    model: sklearn model - trained MultiOutput Classification
    model_filepath: str - path to the model

    OUTPUT
    None
    
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) ==4:
        database_filepath, model_filepath, grid_search = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(grid_search)

        print('Training model...')
        model.fit(X_train, Y_train)
        if grid_search=="True":
            print(model.best_params_)
            joblib.dump(model.best_params_, 'models/best_params.pkl', compress = 1)
        else:
            print("Using tuned model")

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument.'
              'True  to tune model\n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl True')


if __name__ == '__main__':
    main()
