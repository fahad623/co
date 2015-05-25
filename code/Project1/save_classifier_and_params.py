from sklearn.externals import joblib
import os
import shutil
from sklearn.metrics import mean_squared_error
import pre_process
import pandas as pd


def save(clf, X_train, Y_train, bp = None, bs = None):

    pathClassifier = pre_process.clfFolderBase + clf.__class__.__name__ + '/'
    if not os.path.exists(pathClassifier):
        os.makedirs(pathClassifier)

    score_file = open(pathClassifier + "Score.txt", "w")
    if (bp is not None and bs is not None):
        score_file.write("gs.best_params_ = {0}, gs.best_score_ = {1}\n".format(bp, bs))
    
    score_file.write("Classifier Score = {0}\n".format(clf.score(X_train, Y_train)))
    pred = clf.predict(X_train)
    score_file.write("MSE Score = {0}".format(mean_squared_error(Y_train, pred)))
    score_file.close()
    new_df  =pd.DataFrame(pred)
    new_df.to_csv("../../data/pred_train.csv", index = False)
    joblib.dump(clf, pathClassifier+'model.pkl')
