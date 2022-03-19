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
### Pre-processing
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

Querying the unique records for each feature indicated the need to bin the APPLICATION_TYPE and CLASSIFICATION columns. During optimization (below), I wondered if the ASK_AMT was creating some noise in the model, because I was having some difficulty reaching the target accuracy of 75%. I used two different methods to see if the model would improve:
1. Binned the ASK_AMT into two categories - $5000 ask amount and >$5000 ask amount
2. Removed the ASK_AMT from the DataFrame

Neither of the above made any difference to the results; however, I did retain the binned <a href="AlphabetSoupCharity_Optimization_ASK_AMT_grouped.ipynb">ASK_AMT IPYNB Jupyter file</a> for comparison purposes.

Once binning was performed, a OneHotEncoder instance was created and merged to the original application_df DataFrame.

### Compile, Train and Optimize Attempts
Once the data was preprocessed, a train_test_split was performed, using the IS_SUCCESSFUL column as the dependent variable (y) and forty (40) features to be used in the X_train and X_test models. StandardScaler instance was used to scale X_train and y_train. The X_train shape of 25,724 rows and  40 features (columns) was the result.

The resulting IPYNB file for <a href="AlphabetSoupCharity_Optimization.ipynb">AlphabetSoupCharity_Optimization</a> is only one of many iterations of the file required of the research project to optimize the predicatability of the model. Several iterations were conducted to improve the baseline result of **Loss: 0.6250233054161072, Accuracy: 0.679067075252533** created in <a href="AlphabetSoupCharity.ipynb">deliverable one and two</a>. The resulting <a href="AlphabetSoupCharity_Optimization.ipynb">deliverable three</a> was my final four attempts at optimization of the model.

## Results


 
