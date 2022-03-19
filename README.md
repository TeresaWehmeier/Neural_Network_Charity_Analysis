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
Using the charity_data.csv file, the EIN and NAME columns were dropped and the following features (X) were read in for pre-processing
* APPLICATION_TYPE
* AFFILIATION
* CLASSIFICATION
* USE_CASE
* ORGANIZATION
* STATUS
* INCOME_AMT
* SPECIAL_CONSIDERATIONS
* ASK_AMT

The target variable (y) is:
* IS_SUCCESSFUL

Querying the unique records for each feature indicated the need to bin the APPLICATION_TYPE and CLASSIFICATION columns. 

Once binning was performed, a OneHotEncoder instance was created and merged to the original application_df DataFrame.

### Compile, Train and Optimize Attempts
Once the data was preprocessed, a train_test_split was performed, using the IS_SUCCESSFUL column as the dependent variable (y) and forty (40) features to be used in the X_train and X_test models. StandardScaler instance was used to scale X_train and y_train. The X_train shape of 25,724 rows and  40 features (columns) was the result. Finally, the following code was used to save checkpoints at every 5 epochs:
```
import os
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the checkpoint path and filenames
os.makedirs("checkpoints/", exist_ok=True)
checkpoint_path = "checkpoints/weights.{epoch:02d}.hdf5"

# Create a callback that saves the model's weights every 5 epochs
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=4000)
```

## Results
During optimization, I wondered if the ASK_AMT was creating some noise in the model, because over 2/3's of the rows were $5000 (minimum fund request) and very few rows of very large ask amounts (millions). Since I was having some difficulty exceeding 67% accuracy in my neural network models, I used three different methods to see if accuracy would improve:
1. Binned the ASK_AMT into two categories - $5000 ask amount and >$5000 ask amount (saved in separate IPYNB for comparison purposes to final optimization)
2. Binned the ASK_AMT into five bins, grouping the amounts into: 5000 - <50,000; 50,000 - <500,000; 500,000 - <1M, and 1M+ (not saved in IPYNB)
3. Removed the ASK_AMT from the DataFrame (not saved in IPYNB)

Of the three attempts above, the only one that made any difference was #1, so I kept it to compare to unbinned ask amounts<a href="AlphabetSoupCharity_Optimization_ASK_AMT_grouped.ipynb">here</a> for comparison purposes. However, the binning of this column only made a consistent difference in two of my four optimization attempts. The other two made no difference to the accuracy of my model and were not kept.

I also considered removing the STATUS and SPECIAL_CONSIDERATIONS as features in the model, since both had unbalanced distributions; however, leaving these two features out of the model only reduced the amount of data available for testing, and did not improve my results, so I left them in.

The resulting deliverable IPYNB file for<a href="AlphabetSoupCharity_Optimization.ipynb">AlphabetSoupCharity_Optimization</a> is only one of many iterations of the file used during the project to optimize the accuracy of the model. Several iterations were conducted to improve the baseline result of **Loss: 0.6250233054161072, Accuracy: 0.679067075252533** in <a href="AlphabetSoupCharity.ipynb">deliverable one and two</a>. The resulting <a href="AlphabetSoupCharity_Optimization.ipynb">deliverable three</a> were my final four attempts at optimization of the model.



 
