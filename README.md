# Disaster Response Pipeline Project
### Table of Contents


1. [Project Motivation](#overview)
2. [Installation](#installation)
3. [Instructions](#instructions)
4. [File Descriptions](#files)
5. [Discussion](#discussion )
6. [Licensing, Authors, and Acknowledgements](#licensing)

### Project Overview<a name="overview"></a>
## Installation <a name="installation"></a>
The code was tested using Python version 3.9. 
For other necessary libraries please use requirements.txt
```bash
pip install -r requirements.txt
```

### Instructions<a name="instructions"></a>:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        1. In case you with to tune the parameter (GridSearchCV) 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl True`
        2. Otherwise, the model will take for training the optimized parameter
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl False`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions <a name="files"></a>
## Discussion <a name="discussion"></a>
## Licensing, Authors, Acknowledgements!!!!<a name="licensing"></a>


Must give credit to Airbnb.  You can find the Licensing for the data and more useful information  at Airbnb [here](http://insideairbnb.com/get-the-data.html) or at the Kaggle [here](https://www.kaggle.com/airbnb/seattle).
