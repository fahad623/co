import os
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
import numpy as np
import cross_val
import pre_process
import save_classifier_and_params
import _clf_LassoRegression

def make_best_classifier():
    return Ridge(alpha = 6)

def train_base_clf(pp, X_train):
    clf = make_best_classifier()
    alpha_range = np.arange(1, 100, 1)
    param_grid = dict(alpha = alpha_range)
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
    print X_train.shape
    clf, bp, bs = train_base_clf(pp_base, X_train)
