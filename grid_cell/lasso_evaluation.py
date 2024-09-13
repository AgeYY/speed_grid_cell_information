import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from global_setting import RANDOM_STATE

def subset_split(fire_rate, label, sub_sample_size, test_size, random_state=RANDOM_STATE):
    subset_indices = np.random.choice(fire_rate.shape[0], sub_sample_size, replace=False)
    X_subset = fire_rate[subset_indices]
    y_subset = label[subset_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=test_size, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

def lasso_evaluate_multiple_label(fire_rate, label, cv=5, sub_sample_size=2500, test_size=0.2, n_bootstrap=10):
    n_label = label.shape[1]
    r2_list = {}
    for i in range(n_label):
        r2_one_label = lasso_evaluate(fire_rate, label[:, i], cv=cv, sub_sample_size=sub_sample_size, test_size=test_size, n_bootstrap=n_bootstrap)
        r2_list['label' + str(i)] = r2_one_label.copy()
    return r2_list

def lasso_evaluate(fire_rate, label, cv=5, sub_sample_size=2500, test_size=0.2, n_bootstrap=10):
    r2_list = []
    for _ in range(n_bootstrap):
        X_train, X_test, y_train, y_test = subset_split(fire_rate, label, sub_sample_size, test_size, random_state=RANDOM_STATE)
        model = LassoCV(cv=cv).fit(X_train, y_train)  # Lasso with 5-fold cross-validation
        # model = KNeighborsRegressor(n_neighbors=100).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)
    return r2_list
