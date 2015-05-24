from sklearn.ensemble import RandomForestRegressor
import numpy as np
import cross_val
import pre_process
import save_classifier_and_params

def make_best_classifier():
    return RandomForestRegressor(n_estimators = 180)

def train_base_clf(pp_base):    
    clf = make_best_classifier()
    n_estimators_range = np.arange(20, 200, 40)
    param_grid = dict(n_estimators = n_estimators_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp_base.X_train, pp_base.Y_train, param_grid)
    return clf

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase(generate_dummies = True, normalize = False)
    clf, bp, bs = train_base_clf(pp_base)
    save_classifier_and_params.save(clf, pp_base.X_train, pp_base.Y_train, bp, bs)
    

