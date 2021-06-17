# Disaster Response Pipeline Project
## Table of Contents


1. [Project Overview](#overview)
2. [Installation](#installation)
3. [Instructions](#instructions)
4. [File Descriptions](#files)
5. [Discussion](#discussion )
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview<a name="overview"></a>
The purpose of this project is to apply natural language techniques and machine learning in order to classify disaster messages (in English).  Furthermore, there is an API available in which you can enter new messages and automatically receive a classification for them. The model differentiates between 36 categories e.g. “weather”, “water”, “food” etc.
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
        1. In case you wish to tune the parameter (GridSearchCV) 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl True`
        2. Otherwise, the model will take for training the optimized parameter
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl False`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

There you can find the different data visualization for better understanding the data set as well as you can put your message (in English) and receive as a response the classification:

## **1. Visualization** 
![Alt text](images/bar_chart1.png?raw=true)
![Alt text](images/bar_chart2.png?raw=true)
![Alt text](images/pie_chart.png?raw=true)
## **2. Classification**
![Alt text](images/message_analysis1.png?raw=true)
![Alt text](images/message_analysis2.png?raw=true)
## File Descriptions <a name="files"></a>
1.	data

    a) /process_data.py:  ETL pipeline, clean, preprocess data and store it into SQLite database

    b) /disaster_messages.csv: real messages that were sent during disaster event

    c) /disaster_categories.csv: 36 possible categories

2.	model

    a) /train_classifier.py: takes data from database, creates and trains, tunes classifier and at the end saves data to pickle file

    b) /best_params.pkl : contains tuned parameter
3.	app

    a) /run.py: runs REST API for visualizations and message classification 

    b) /wrangling_script/ wrangle_data.py:  data wrangling as well as data visualization

    c) /templates: all files needed for frontend
   

## Discussion <a name="discussion"></a>
Due to the fact, that “Disaster Response Project” is a supervised learning with 36 pre-defined categories we are dealing with classification task. In this project **K-nearest neighbors** will be used as classification algorithm. 

**K-nearest neighbors** belongs to the type of a lazy learner that means that this algorithm doesn’t learn a discriminative function from the training set instead it memorizes the training data. The major advantage of this approach on the one hand is the instant adoption of new data points. On the other hand, this approach requires high computational cost especially for classifying new samples (the computation complexity grows linearly with the number of samples in the data set). In addition, K-nearest neighbors demands high storage space since the training samples can’t be discard (training isn’t a part of the approach).

The main idea of the K-nearest neighbors is:
1.	Define _k_ and a distance metric
2.	Find the k-nearest neighbors for new sample
3.	Classify the sample by majority vote

For this reason, there are two parameters that are crucial to balance between overfitting and underfitting: the number of _k_ and the _distance metric_. Therefore, these two parameters are used for tuning. 

The most popular distance metrics are:
1.	**Euclidean Distance**: the straight line between two data points in Euclidean space

     <a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{x_2-x_1)^2&plus;(y_2-y_1)^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{x_2-x_1)^2&plus;(y_2-y_1)^2}" title="\sqrt{x_2-x_1)^2+(y_2-y_1)^2}" /></a>

2.	**Manhattan Distance**:
the distance between two points or vectors A and B is defined as the sum of the absolute differences of their individual coordinates 

    <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i}|A_i-B_i|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i}|A_i-B_i|" title="\sum_{i}|A_i-B_i|" /></a>

3.	**Minkowski Distance**: a generalization of the Euclidean and Manhattan distance 

    <a href="https://www.codecogs.com/eqnedit.php?latex=(\sum_{i=1}^{n}|x_i-y_i|^p)^&space;\frac{1}{p}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\sum_{i=1}^{n}|x_i-y_i|^p)^&space;\frac{1}{p}" title="(\sum_{i=1}^{n}|x_i-y_i|^p)^ \frac{1}{p}" /></a>

## Licensing, Authors, Acknowledgements<a name="licensing"></a>


Must give credit to [Figure Eight](https://appen.com/) that provided the dataset with real messages and labels. Great thanks to Udacity for their contribution during the process. 



###### Copyright (C) 2021 June
###### TO THE FULLEST EXTENT PERMITTED UNDER APPLICABLE LAW, THE CODE COMPONENTS ARE PROVIDED BY THE AUTHORS, COPYRIGHT HOLDERS, CONTRIBUTORS, LICENSORS, “AS IS”.

######  DISCLAIMED ARE ANY REPRESENTATIONS OR WARRANTIES OF ANY KIND, WHETHER ORAL OR WRITTEN, WHETHER EXPRESS, IMPLIED, OR ARISING BY STATUTE, CUSTOM, COURSE OF DEALING, OR TRADE USAGE, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
######  IN NO EVENT WILL THE COPYRIGHT OWNER, CONTRIBUTORS, LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION). HOWEVER, CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE CODE COMPONENTS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
