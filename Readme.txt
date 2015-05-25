
Project1 -
How to run - 

1. Please copy the dataset to the 'data' folder
2. The development environment is latest Anaconda distribution(with Python 2.7)
3. Run 'code/Project1/final_model.py'
4. Predictions are written to the 'output' folder


Explanation of the model building process - 
1. Separate the data into numerical and categorial variables
2. For the numerical variables missing values were imputed with the mean since there are large number of examples with missing data and we cannot just exclude them. If the meaning of the features is known, then we can use other techniques to impute the missing values.
3. Perform feature scaling on numerical variables (mean normalization)
4. Convert categorical variable into dummy/indicator variables        
5. Then combine them again to get the final train, test set
6. Run classifiers and test their accuracy using 10 fold cross validation
7. Save the classifiers and scores in the 'classifier' folder
8. Several different classifiers were tried and in the end we picked Lasso as stage1 classifer. This lets us pick the relevant features (reduced to 58 from 255) and then train stage2 classifier(SVR) on the reduced features.
8. When 'final_model.py' is run for the first time, it will save the trained classifiers on disk, so that subsequent runs are faster
9. The results are saved in 'output' folder


Project2 -

Open the ipython notebook from the 'ipython' folder.