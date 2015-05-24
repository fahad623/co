
Project1 -
How to run - 

1. Please create a folder 'data' at the same level as code and copy the dataset to it
2. Run '.py'


Explanation of the model building process - 
1. Separate the data into numerical and categorial variables
2. Perform feature scaling on numerical variables
3. Convert categorical variable into dummy/indicator variables        
4. Then combine them again to get the final train, test set
5. Run classifiers and test their accuracy using 10 fold cross validation
6. Save the classifiers and scores in the 'classifier' folder at the same level as code
7. Pick the best classifer, load it from disk and output the predictions on test set
8. The results are saved in 'output' folder