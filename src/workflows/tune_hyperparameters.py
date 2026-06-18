from sklearn.metrics import pairwise_distances, cohen_kappa_score
import numpy as np
from sklearn.svm import OneClassSVM
from src.models.monthly_model.monthly_climademic_model import ClimademicMonthlyModel
from sklearn.model_selection import GridSearchCV

#calculate median square pairwise distance between features. Used in defining gamma parameter
def calculate_median(X):
    distance_d2 = pairwise_distances(X, metric="sqeuclidean")
    median_d2 = np.median(distance_d2[distance_d2 > 0])
    return median_d2

#perform grid search scoring against the accuracy
def grid_search_params(X_train,y_train, X_test, y_test):

    #define grid for parameter search for nu
    nu_grid = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    
    gamma_multipliers = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])
    gamma_init = 1 / (2 * calculate_median(X_train)) #choose initial gamma based on median heuristic

    #define grid for parameter search for gamma. Based around median heuristic
    gamma_grid = gamma_init * gamma_multipliers
    
    clf=OneClassSVM()
    parameters=[{'nu':nu_grid,'kernel':['rbf'],'gamma':gamma_grid}]
    grid_search=GridSearchCV(estimator=clf,param_grid=parameters,scoring='accuracy',cv=5,verbose=0)
    grid_search.fit(X_train,y_train)

    print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

    print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

    print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

    print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))

    return grid_search.best_params_


#perform grid search scoring against the best agreements
def _grid_search_params(X):
    X = np.array(X)
    probability_threshold = 0.5
    nu_grid = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    
    gamma_multipliers = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])
    gamma_init = 1 / (2 * calculate_median(X))

    gamma_grid = gamma_init * gamma_multipliers
    #print(gamma_init, gamma_grid)
    results = []
    n_splits = 5
    random_state = 4
    rng = np.random.default_rng(random_state)
    for nu in nu_grid:
        for gamma in gamma_grid:
            kappas = []
            params = f"-s 2 -t 2 -g {gamma} -n {nu} -b 1 -q"
            print(params)
            for _ in range(n_splits):
                perm = rng.permutation(len(X))
                n_test = int(0.3 * len(X))
                n_train = (len(X) - n_test) // 2
                
                test_idx = perm[:n_test]
                train_idx_1 = perm[n_test:n_test + n_train]
                train_idx_2 = perm[n_test + n_train:]
                 
                X_train_1 = X[train_idx_1].tolist()
                X_train_2 = X[train_idx_2].tolist()
                X_test = X[test_idx].tolist()

                y_train_1 = [1] * len(X_train_1)
                y_train_2 = [1] * len(X_train_2)

                m1 = ClimademicMonthlyModel(params=params)
                m2 = ClimademicMonthlyModel(params=params)

                m1.train(X_train_1, y_train_1)
                m2.train(X_train_2, y_train_2)

                labels1, acc1, values1 = m1.predict(X_test, probability=True)
                labels2, acc2, values2 = m2.predict(X_test, probability=True)

                p1 = np.array(values1)[:, 0]
                p2 = np.array(values2)[:, 0]

                y1 = p1 >= probability_threshold
                y2 = p2 >= probability_threshold

                if len(np.unique(y1)) < 2 and len(np.unique(y2)) < 2:
                    kappa = 1.0 if np.all(y1 == y2) else 0.0
                else:
                    kappa = cohen_kappa_score(y1, y2)
                
                if not np.isnan(kappa):
                    kappas.append(kappa)

            results.append({
                "nu": nu,
                "gamma": gamma,
                "gamma_multiplier": gamma / gamma_init,
                "mean_kappa": float(np.mean(kappas)) if kappas else np.nan,
                "std_kappa": float(np.std(kappas)) if kappas else np.nan,
                "gamma0": gamma_init,
                #"median_d2": median_d2,
                "params": params,
                })

    results = sorted(
        results,
        key=lambda d: (
            -d["mean_kappa"],
            d["nu"],
            abs(np.log(d["gamma_multiplier"]))
        )
    )

    print(results[0])

    return results[0], results



