import numpy as np
import pandas as pd
import pre_process
from sklearn.externals import joblib
import _clf_LassoRegression
import _clf_LassoAndSVR
import os
import shutil

if __name__ == '__main__':

    #Preprocess the training file
    pp_base = pre_process.PreProcessBase()
    X_train, Y_train = pp_base.get_train()

    #Preprocess the test file
    pp_base.clean_test(pre_process.testFile)
    X_test = pp_base.get_test()
    
    #Fit or load Stage1 classifier
    pathClassifier = pre_process.clfFolderBase + 'Lasso/'
    if os.path.exists(pathClassifier):
        clf_stage1 = joblib.load(pathClassifier+'model.pkl')
    else:
        os.makedirs(pathClassifier)
        clf_stage1 = _clf_LassoRegression.make_best_classifier()  #Best params selected before by CV
        clf_stage1.fit(X_train, Y_train)
        joblib.dump(clf_stage1, pathClassifier+'model.pkl')

    #Select the relevant features
    X_train = pp_base.X_train[:, clf_stage1.coef_ != 0]
    print clf_stage1.coef_[clf_stage1.coef_ != 0].shape
    print X_train.shape

    X_test = pp_base.X_test[:, clf_stage1.coef_ != 0]

    #Fit or load Stage2 classifier
    pathClassifier = pre_process.clfFolderBase + 'SVR/' 
    if os.path.exists(pathClassifier):
        clf_stage2 = joblib.load(pathClassifier+'model.pkl')
    else:
        os.makedirs(pathClassifier)
        clf_stage2 = _clf_LassoAndSVR.make_best_classifier() #Best params selected before by CV
        clf_stage2.fit(X_train, Y_train)
        joblib.dump(clf_stage2, pathClassifier+'model.pkl')

    #Predict on the test set and save the output
    pred = clf_stage2.predict(X_test)
    out_df = pd.DataFrame(pred)

    if not os.path.exists(pre_process.outFolder):
        os.makedirs(pre_process.outFolder)

    out_df.to_csv(pre_process.outFolder + "pred_test.txt", index = False, header = False)
    

    