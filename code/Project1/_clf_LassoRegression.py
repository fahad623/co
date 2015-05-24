from sklearn.linear_model import Lasso
import numpy as np
import cross_val
import pre_process
import save_classifier_and_params

def make_best_classifier():
    return Lasso(alpha = 0.1)

def train_base_clf(pp):
    clf = make_best_classifier()
    alpha_range = np.logspace(-4, 3, num=8)
    param_grid = dict(alpha = alpha_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    return clf, bp, bs

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf, bp, bs = train_base_clf(pp_base)
    save_classifier_and_params.save(clf, pp_base.X_train, pp_base.Y_train, bp, bs)