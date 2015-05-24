import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
trainFile = "../../data/codetest_train_trunc.txt"
#testFile = "../../data/codetest_test.txt"
outFile = "../../data/codetest_train_out.csv"


#df_train = pd.read_csv(trainFile, sep = '\t')
#df_test= pd.read_csv(testFile, sep = '\t')


#print df_train.describe()

#from sklearn import preprocessing
#import numpy as np
#X = np.array([[ 1., np.nan,  2.],
#              [ 2.,  0.,  0.],
#              [ 0.,  1., -1.]])

#scaler = preprocessing.StandardScaler().fit(X)
#print scaler

df_train = pd.read_csv(trainFile, sep='\t')
#d = df.T.to_dict().values()
#print d

#vec = DictVectorizer()
#x = vec.fit_transform(d)
#print x.toarray()
#print vec.get_feature_names()


#def add_dummies(indata):
#    df_nums = indata.select_dtypes(include=['floating']).fillna(0.0)    
#    df_cats = indata.select_dtypes(exclude=['floating'])
#    dummies = pd.get_dummies(df_cats, dummy_na = True)
#    outdata = pd.concat([df_nums, dummies], axis=1)
#    return outdata

#out = add_dummies(df_train)
#out.to_csv(outFile, index = False) 
print np.logspace(-5, -1, num=5)

arr1 = np.array([[1.0, 2, 4], [3, 4, 6]])

arr2 = np.array([1, 0.0, 0])

arr3 = arr1[:, arr2 != 0]

print arr1
print arr2
print arr3