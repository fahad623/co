from sklearn.svm import SVR
import os
from sklearn.externals import joblib
import numpy as np
import cross_val
import pre_process
import save_classifier_and_params
import _clf_LassoRegression

def make_best_classifier():
    return SVR(C = 10.0, kernel = 'rbf')

def train_base_clf(pp, X_train):
    clf = make_best_classifier()
    C_range = np.logspace(-3, 1, num=5)
    epsilon_range = np.logspace(-5, -1, num=5)
    gamma_range = np.logspace(-2, 2, num = 5)
    #C_range = np.arange(8, 14, 1)
    param_grid = dict(C = C_range, epsilon = epsilon_range, gamma = gamma_range, kernel = ['linear', 'rbf'])
    
    clf, bp, bs = cross_val.fit_clf(clf, X_train, pp.Y_train, param_grid)
    return clf, bp, bs

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()

    pathClassifier = pre_process.clfFolderBase + 'Lasso/'
    if os.path.exists(pathClassifier):
        clf_stage1 = joblib.load(pathClassifier+'model.pkl')
    else:
        clf_stage1 = _clf_LassoRegression.make_best_classifier()
        clf_stage1.fit(pp_base.X_train, pp_base.Y_train)

    X_train = pp_base.X_train[:, clf_stage1.coef_ != 0]
    clf, bp, bs = train_base_clf(pp_base, X_train)

    print clf.support_.shape
    
    save_classifier_and_params.save(clf, X_train, pp_base.Y_train, bp, bs)