'''Model CA Wildfire data

Functions:
- metrics_reg
- baseline
- rfe_rev
- reg_mods
- final_models
- test_model
- plt_err
'''
########## IMPORTS ##########

import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE

######### FUNCTIONS #########

def metrics_reg(y, y_pred):
    """
    Input y and y_pred & get RMSE, R2
    """
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    return round(rmse,2), round(r2,4)

def baseline(ytr,yv):
    """
    The function calculates and prints the baseline metrics of a model
    that always predicts the mean in the target variable.
    """
    pred_mean = ytr.mean()[0]
    ytr_p = ytr.assign(pred_mean=pred_mean)
    yv_p = yv.assign(pred_mean=pred_mean)
    rmse_tr = mean_squared_error(ytr,ytr_p.pred_mean)**.5
    rmse_v = mean_squared_error(yv,yv_p.pred_mean)**.5
    r2_tr = r2_score(ytr, ytr_p.pred_mean)
    r2_v = r2_score(yv, yv_p.pred_mean)
    print(f'Baseline Fire Size: {round(((ytr.fire_size).mean()),2)}')
    print(f'Train       RMSE: {rmse_tr}   R2: {r2_tr}')
    print(f'Validate    RMSE: {rmse_v}    R2: {r2_v}')

def rfe_rev(Xs_train,y_train,r):
    '''Get RFE ranks in a dataframe'''
    lr = LinearRegression()
    rfe = RFE(lr,n_features_to_select=r)
    rfe.fit(Xs_train,y_train)
    rfe_ranks_df = pd.DataFrame({'Var':Xs_train.columns.to_list(),'Rank':rfe.ranking_})
    return rfe_ranks_df.sort_values('Rank')

def reg_mods(Xtr,ytr,Xv,yv,features=None):
    '''
    Input X_train,y_train,X_val,y_val, list of features, and alpha, degree, and power
    so that function will run through linear regression, lasso lars,
    polynomial feature regression, and tweedie regressor (glm)
    - diff feature combos
    - diff hyper params
    - output as df
    '''
    if features is None:
        features = Xtr.columns.to_list()
    # baseline as mean
    pred_mean = ytr.mean()[0]
    ytr_p = ytr.assign(pred_mean=pred_mean)
    yv_p = yv.assign(pred_mean=pred_mean)
    rmse_tr = mean_squared_error(ytr,ytr_p.pred_mean)**.5
    rmse_v = mean_squared_error(yv,yv_p.pred_mean)**.5
    r2_tr = r2_score(ytr, ytr_p.pred_mean)
    r2_v = r2_score(yv, yv_p.pred_mean)
    output = {
            'model':'bl_mean',
            'features':'None',
            'params':'None',
            'rmse_tr':rmse_tr,
            'rmse_v':rmse_v,
            'r2_tr':r2_tr,
            'r2_v':r2_v
        }
    metrics = [output]
    # create iterable for feature combos
    for r in range(1,(len(features)+1)):
        # cycle through feature combos for linear reg
        for feature in itertools.combinations(features,r):
            f = list(feature)
            # linear regression
            lr = LinearRegression()
            lr.fit(Xtr[f],ytr)
            # metrics
            pred_lr_tr = lr.predict(Xtr[f])
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lr_tr)
            pred_lr_v = lr.predict(Xv[f])
            rmse_v,r2_v = metrics_reg(yv,pred_lr_v)
            # table-ize
            output ={
                    'model':'LinearRegression',
                    'features':f,
                    'params':'None',
                    'rmse_tr':rmse_tr,
                    'rmse_v':rmse_v,
                    'r2_tr':r2_tr,
                    'r2_v':r2_v
                }
            metrics.append(output)
        # cycle through feature combos and alphas for lasso lars
        for feature in itertools.combinations(features,r):
            f = list(feature)
            # lasso lars
            ll = LassoLars(random_state=42)
            ll.fit(Xtr[f],ytr)
            # metrics
            pred_ll_tr = ll.predict(Xtr[f])
            rmse_tr,r2_tr = metrics_reg(ytr,pred_ll_tr)
            pred_ll_v = ll.predict(Xv[f])
            rmse_v,r2_v = metrics_reg(yv,pred_ll_v)
            # table-ize
            output ={
                    'model':'LassoLars',
                    'features':f,
                    'params':'None',
                    'rmse_tr':rmse_tr,
                    'rmse_v':rmse_v,
                    'r2_tr':r2_tr,
                    'r2_v':r2_v
                }
            metrics.append(output)
        # cycle through feature combos and degrees for polynomial feature reg
        for feature in itertools.combinations(features,r):
            f = list(feature)
            # polynomial feature regression
            pf = PolynomialFeatures()
            Xtr_pf = pf.fit_transform(Xtr[f])
            Xv_pf = pf.transform(Xv[f])
            lp = LinearRegression()
            lp.fit(Xtr_pf,ytr)
            # metrics
            pred_lp_tr = lp.predict(Xtr_pf)
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lp_tr)
            pred_lp_v = lp.predict(Xv_pf)
            rmse_v,r2_v = metrics_reg(yv,pred_lp_v)
            # table-ize
            output ={
                    'model':'PolynomialFeature',
                    'features':f,
                    'params':'None',
                    'rmse_tr':rmse_tr,
                    'rmse_v':rmse_v,
                    'r2_tr':r2_tr,
                    'r2_v':r2_v
                }
            metrics.append(output)
        # cycle through feature combos, alphas, and powers for tweedie reg
        for feature in itertools.combinations(features,r):
            f = list(feature)
            # tweedie regressor glm
            lt = TweedieRegressor(power=2)
            lt.fit(Xtr[f],ytr.fire_size)
            # metrics
            pred_lt_tr = lt.predict(Xtr[f])
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lt_tr)
            pred_lt_v = lt.predict(Xv[f])
            rmse_v,r2_v = metrics_reg(yv,pred_lt_v)
            # table-ize
            output ={
                    'model':'TweedieRegressor',
                    'features':f,
                    'params':'power=2',
                    'rmse_tr':rmse_tr,
                    'rmse_v':rmse_v,
                    'r2_tr':r2_tr,
                    'r2_v':r2_v
                }
            metrics.append(output)
    return pd.DataFrame(metrics)

def final_model(model,X_train,y_train,X_val,y_val):
    '''Input model type along with train and validate data and
    it will return RMSE and R2 results per the selected model
    
    Please include model argument: lr, poly, lasso, tweedie'''
    if model == 'lr':
        # features
        f=['lat_s', 'long_s', 'elevation_mean_s', 'percent_invasive_plant_s', 'temp_mean_s', 'humidity_mean_s', 'most_common_is_hardwood_s']
        # model
        lr = LinearRegression()
        lr.fit(X_train[f],y_train)
        # metrics
        pred_lr_tr = lr.predict(X_train[f])
        rmse_tr,r2_tr = metrics_reg(y_train,pred_lr_tr)
        pred_lr_v = lr.predict(X_val[f])
        rmse_v,r2_v = metrics_reg(y_val,pred_lr_v)
        print('Linear Regression')
        print(f'Train       RMSE: {rmse_tr}   R2: {r2_tr}')
        print(f'Validate    RMSE: {rmse_v}    R2: {r2_v}')
    elif model == 'poly':
        # features
        f=['lat_s', 'long_s', 'percent_invasive_plant_s', 'temp_mean_s', 'humidity_mean_s', 'wind_speed_mean_s', 'most_common_is_hardwood_s']
        # polynomial feature regression
        pf = PolynomialFeatures(degree=2)
        X_train_pf = pf.fit_transform(X_train[f])
        X_val_pf = pf.transform(X_val[f])
        # model
        pr = LinearRegression()
        pr.fit(X_train_pf,y_train)
        # metrics
        pred_pr_tr = pr.predict(X_train_pf)
        rmse_tr,r2_tr = metrics_reg(y_train,pred_pr_tr)
        pred_pr_v = pr.predict(X_val_pf)
        rmse_v,r2_v = metrics_reg(y_val,pred_pr_v)
        print('Polynomial Features through Linear Regression')
        print(f'Train       RMSE: {rmse_tr}   R2: {r2_tr}')
        print(f'Validate    RMSE: {rmse_v}    R2: {r2_v}')
    elif model == 'lasso':
        # features
        f=['temp_mean_s', 'humidity_mean_s']
        # model
        ll = LassoLars()
        ll.fit(X_train[f],y_train)
        # metrics
        pred_ll_tr = ll.predict(X_train[f])
        rmse_tr,r2_tr = metrics_reg(y_train,pred_ll_tr)
        pred_ll_v = ll.predict(X_val[f])
        rmse_v,r2_v = metrics_reg(y_val,pred_ll_v)
        print('Lasso Lars')
        print(f'Train       RMSE: {rmse_tr}   R2: {r2_tr}')
        print(f'Validate    RMSE: {rmse_v}    R2: {r2_v}')
    elif model == 'tweedie':
        # features
        f=['long_s', 'trees_per_acre_mean_s', 'percent_invasive_plant_s', 'temp_mean_s', 'humidity_mean_s', 'most_common_is_hardwood_s']
        # model
        ll = TweedieRegressor(power=2)
        ll.fit(X_train[f],y_train.fire_size)
        # metrics
        pred_ll_tr = ll.predict(X_train[f])
        rmse_tr,r2_tr = metrics_reg(y_train,pred_ll_tr)
        pred_ll_v = ll.predict(X_val[f])
        rmse_v,r2_v = metrics_reg(y_val,pred_ll_v)
        print('Lasso Lars')
        print(f'Train       RMSE: {rmse_tr}   R2: {r2_tr}')
        print(f'Validate    RMSE: {rmse_v}    R2: {r2_v}')
    else:
        print('Please include model argument: lr, poly, lasso, tweedie')

def test_model(X_train,y_train,X_test,y_test):
    '''Input train and test data and it will return RMSE and R2 test results'''
    # features
    f=['lat_s', 'long_s', 'percent_invasive_plant_s', 'temp_mean_s', 'humidity_mean_s', 'wind_speed_mean_s', 'most_common_is_hardwood_s']
    # polynomial feature regression
    pf = PolynomialFeatures(degree=2)
    X_train_pf = pf.fit_transform(X_train[f])
    X_test_pf = pf.transform(X_test[f])
    # model
    pr = LinearRegression()
    pr.fit(X_train_pf,y_train)
    # metrics
    pred_pr_t = pr.predict(X_test_pf)
    rmse_t,r2_t = metrics_reg(y_test,pred_pr_t)
    print('Polynomial Features through Linear Regression')
    print(f'Test    RMSE: {rmse_t}    R2: {r2_t}')

def plt_err(Xs_train,y_train,Xs_test,y_test):
    '''plot predicted vs actual property values by inputting train and test'''
    # features
    f=['lat_s', 'long_s', 'percent_invasive_plant_s', 'temp_mean_s', 'humidity_mean_s', 'wind_speed_mean_s', 'most_common_is_hardwood_s']
    # polynomial feature regression
    pf = PolynomialFeatures(degree=2)
    X_train_pf = pf.fit_transform(Xs_train[f])
    X_test_pf = pf.transform(Xs_test[f])
    # model
    pr = LinearRegression()
    pr.fit(X_train_pf,y_train)
    # metrics
    pred_pr_t = pd.DataFrame(pr.predict(X_test_pf),index=y_test.index,columns=['y_pred'])
    pred_mean = y_test
    pred_mean = pred_mean.assign(baseline=pred_mean.fire_size.mean())
    plt.figure(figsize=(16,8))
    plt.plot(y_test, pred_mean.baseline, alpha=.5, color="black", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (680, 40))
    plt.scatter(y_test, y_test, alpha=.5, color="red", label='The Ideal Line: Predicted = Actual')
    plt.annotate("The Ideal Line: Predicted = Actual", (650, 600), rotation=26)
    plt.scatter(y_test, pred_pr_t, alpha=.1, color="orange", s=100, label="2nd degree Polynomial Predictions")
    plt.legend(loc='upper center')
    plt.xlabel("Actual Wildfire Size (Acres)")
    plt.ylabel("Predicted Wildfire Size (Acres)")
    plt.title("Did I predict baseline by accident? \nOh wait, the model just isn't that good yet")
    plt.show()
