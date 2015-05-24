from sklearn.linear_model import LinearRegression
import pre_process
import save_classifier_and_params


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf = LinearRegression()
    clf.fit(pp_base.X_train, pp_base.Y_train)
    save_classifier_and_params.save(clf, pp_base.X_train, pp_base.Y_train)
    
    

