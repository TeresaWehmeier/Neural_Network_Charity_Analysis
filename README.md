# Neural Network Charity Analysis

## Overview
Alphabet Soup, a hypothetical charity foundation, needs to develop a method of predicting applicants that will be succesful if funded by the charity. The research department has been asked to develop a neural network and create a binary classifier capable of making predictions based on the applicant data.

### Resources:
* Sklearn.model_selection: train_test_split
* Sklearn.preprocessing: StandardScaler
* Sklearn.ensemble: RandomForestClassifier
* Sklearn.metrics: accuracy_score
* Sklearn.preprocessing: OneHotEncoder
* Pandas
* Tensorflow
* Data: charity_data.csv
* Applications: Python 3.9, Jupyter Notebook

## Development Process
Using the charity_data.csv file, the EIN and NAME columns were dropped and the following features were read in for pre-processing
* APPLICATION_TYPE
* AFFILIATION
* CLASSIFICATION
* USE_CASE
* ORGANIZATION
* STATUS
* INCOME_AMT
* SPECIAL_CONSIDERATIONS
* ASK_AMT
* IS_SUCCESSFUL

Querying the unique records for each feature indicated the need to bin the APPLICATION_TYPE and CLASSIFICATION columns. Once binning was performed OneHotEncoder was used to create
