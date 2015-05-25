import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

trainFile = "../../data/codetest_train.txt"
testFile = "../../data/codetest_test.txt"
clfFolderBase = "../../classifier/"
outFolder = "../../output/"

class PreProcessBase(object):   

    def __init__(self, generate_dummies = True, normalize = True):
        self.normalize = normalize
        self.generate_dummies = generate_dummies
        self.load()

    def clean_test(self, testFile):
        df_test = pd.read_csv(testFile, sep ='\t')

        df_nums_test = df_test.select_dtypes(include=['floating'])  
        df_cats_test = df_test.select_dtypes(exclude=['floating'])
        test_nums = df_nums_test.values

        test_nums = self.imp.transform(test_nums);  

        if self.normalize:
            test_nums = self.scaler.transform(test_nums)

        if (self.generate_dummies):
            dummies = pd.get_dummies(df_cats_test, dummy_na = True)
            self.X_test = np.hstack((test_nums, dummies.values))
        else:
            self.X_test = np.hstack((test_nums, df_cats_test.values)) 

        del df_test
     
    def clean_train(self, df_train):

        #Separate the data into numerical and categorial variables
        #Impute missing numerical values with the mean
        #Perform feature scaling on numerical variables
        #Convert categorical variable into dummy/indicator variables        
        #Then combine them again to get the final train set

        df_nums_train = df_train.ix[:, 1:].select_dtypes(include=['floating'])
        df_cats_train = df_train.ix[:, 1:].select_dtypes(exclude=['floating'])        

        train_nums = df_nums_train.values        

        self.imp = Imputer()
        self.imp.fit(train_nums);     
        train_nums = self.imp.transform(train_nums); 
        if self.normalize:
            train_nums = self.normalize_data(train_nums)
            new_df  =pd.DataFrame(train_nums)

        if (self.generate_dummies):
            dummies = pd.get_dummies(df_cats_train, dummy_na = True)
            self.X_train = np.hstack((train_nums, dummies.values))
        else:
            self.X_train = np.hstack((train_nums, df_cats_train.values))

    def normalize_data(self, train):
        self.scaler = preprocessing.StandardScaler()
        train_nums = self.scaler.fit_transform(train)        
        return train_nums

    def load(self):
        df_train = pd.read_csv(trainFile, sep ='\t')
        self.clean_train(df_train)      
        self.Y_train = df_train.ix[:, 0].values

        del df_train

    def get_train(self):
        return self.X_train, self.Y_train

    def get_test(self):
        self.X_test

if __name__ == '__main__':
    pp_base = PreProcessBase()
    pp_base.clean_test(testFile)

