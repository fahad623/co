import pandas as pd
import numpy as np
from sklearn import preprocessing

trainFile = "../../data/codetest_train.txt"
testFile = "../../data/codetest_test.txt"

class PreProcessBase(object):   

    def __init__(self, generate_dummies = True, normalize = True):
        self.normalize = normalize
        self.generate_dummies = generate_dummies
        self.load()
     
    def clean(self, df_train, df_test):

        #Separate the data into numerical and categorial variables
        #Perform feature scaling on numerical variables
        #Convert categorical variable into dummy/indicator variables        
        #Then combine them again to get the final train, test set

        df_nums_train = df_train.ix[:, 1:].select_dtypes(include=['floating']).fillna(0.0)    
        df_cats_train = df_train.ix[:, 1:].select_dtypes(exclude=['floating'])

        df_nums_test = df_test.select_dtypes(include=['floating']).fillna(0.0)    
        df_cats_test = df_test.select_dtypes(exclude=['floating'])

        train_nums = df_nums_train.values
        test_nums = df_nums_test.values

        if self.normalize:
            train_nums, test_nums = self.normalize_data(train_nums, test_nums)

        if (self.generate_dummies):
            dummies = pd.get_dummies(df_cats_train, dummy_na = True)
            self.X_train = np.hstack((train_nums, dummies.values))

            dummies = pd.get_dummies(df_cats_test, dummy_na = True)
            self.X_test = np.hstack((test_nums, dummies.values))
        else:
            self.X_train = np.hstack((train_nums, df_cats_train.values))
            self.X_test = np.hstack((test_nums, df_cats_test.values))
        
        

    def normalize_data(self, train, test):
        scaler = preprocessing.StandardScaler()
        train_nums = scaler.fit_transform(train)
        test_nums = scaler.transform(test)
        return train_nums, test_nums

    def load(self):
        df_train = pd.read_csv(trainFile, sep ='\t')
        df_test = pd.read_csv(testFile, sep ='\t')

        self.clean(df_train, df_test)      
        self.Y_train = df_train.ix[:, 0].values

        del df_train, df_test

    def get_train_test(self):
        return self.X_train, self.X_test, self.Y_train

