from sklearn.linear_model import LinearRegression
import numpy as np
import cross_val
import pre_process
import save_classifier_and_params

def make_best_classifier():
    return LinearRegression()

def train_base_clf(pp):
    clf = make_best_classifier()
    param_grid = dict()
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    return clf, bp, bs

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase(normalize = False, generate_dummies = False)
    clf, bp, bs = train_base_clf(pp_base)
    print clf.coef_
    save_classifier_and_params.save(clf, pp_base.X_train, pp_base.Y_train, bp, bs)
    
    

