from sklearn.svm import SVR
import numpy as np
import cross_val
import pre_process
import save_classifier_and_params

def make_best_classifier():
    return SVR(C = 0.1, gamma = 1e-5, kernel = 'linear')

def train_base_clf(pp):
    clf = make_best_classifier()
    C_range = np.logspace(-4, 0, num=5)
    gamma_range = np.logspace(-5, -1, num=5)
    #C_range = np.arange(0.01, 0.02, 0.01)
    param_grid = dict(C = C_range, gamma = gamma_range, kernel = ['linear', 'poly','rbf'])
    
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    return clf, bp, bs

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf, bp, bs = train_base_clf(pp_base)
    print clf.support_.shape
    save_classifier_and_params.save(clf, pp_base.X_train, pp_base.Y_train, bp, bs)