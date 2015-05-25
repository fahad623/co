from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import numpy as np


def cv_optimize(clf, X_train, Y_train, param_grid):
    gs = GridSearchCV(clf, param_grid = param_grid, scoring = 'mean_squared_error', cv = KFold(X_train.shape[0], n_folds = 200, shuffle = True, random_state = 78), n_jobs = 6, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def fit_clf(clf, X_train, Y_train, param_grid = dict()):
    clf, bp, bs = cv_optimize(clf, X_train, Y_train, param_grid)    
    clf.fit(X_train, Y_train)
    return clf, bp, bs