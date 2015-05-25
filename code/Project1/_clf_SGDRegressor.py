from sklearn.linear_model import SGDRegressor
import numpy as np
import cross_val
import pre_process
import save_classifier_and_params

def make_best_classifier():
    return SGDRegressor(alpha = 100)

def train_base_clf(pp):
    clf = make_best_classifier()
    alpha_range = np.logspace(-6, 1, num=8)
    l1_ratio_range = np.arange(0.01, 0.5, 0.03)
    n_iter_range = np.arange(5, 20, 1)
    param_grid = dict(alpha = alpha_range, n_iter = n_iter_range, l1_ratio = l1_ratio_range, penalty = ['l2' ,'elasticnet'])
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    return clf, bp, bs

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf, bp, bs = train_base_clf(pp_base)

    print clf.coef_[clf.coef_ != 0].shape
    save_classifier_and_params.save(clf, pp_base.X_train, pp_base.Y_train, bp, bs)