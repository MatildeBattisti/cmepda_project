# XGBoost has to be imported before ROOT to avoid crashes because of clashing
#import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import uproot
import time
import json

"""
    Reads data from ROOT file and selects ML alorithm's parameters
"""
def read_data():
    file = uproot.open("datasets/skimmed2.root")
    tree = file["Events"]

    params = [
        "nJet",
        "Jet_eta_bst",
        "Jet_pt_bst",
        "Jet_phi_bst",
        "Jet_mass_bst",
        "Jet_eta_bst_log",
        "Jet_pt_bst_log",
        "Jet_phi_bst_log",
        "Jet_mass_bst_log",
        "Jet_eta_bnd",
        "Jet_pt_bnd",
        "Jet_phi_bnd",
        "Jet_mass_bnd",
        "Jet_eta_bnd_log",
        "Jet_pt_bnd_log",
        "Jet_phi_bnd_log",
        "Jet_mass_bnd_log",
        "Jet_eta_brd",
        "Jet_pt_brd",
        "Jet_phi_brd",
        "Jet_mass_brd",
        "Jet_eta_brd_log",
        "Jet_pt_brd_log",
        "Jet_phi_brd_log",
        "Jet_mass_brd_log",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        "MET_phi",
        "MET_pt",
        "MET_significance",
        "MET_pt_log",
        "m_hh"
    ]
    target = ["GenMET_pt"]

    data = tree.arrays(params + target, library="np")

    print(f'✅ Read data from .root file')
    return params, target, data

def data_engineering(data):
    npFeatures = np.array([
        data["nJet"],
        data["Jet_eta_bst"],
        data["Jet_pt_bst"],
        data["Jet_phi_bst"],
        data["Jet_mass_bst"],
        data["Jet_eta_bst_log"],
        data["Jet_pt_bst_log"],
        data["Jet_phi_bst_log"],
        data["Jet_mass_bst_log"],
        data["Jet_eta_bnd"],
        data["Jet_pt_bnd"],
        data["Jet_phi_bnd"],
        data["Jet_mass_bnd"],
        data["Jet_eta_bnd_log"],
        data["Jet_pt_bnd_log"],
        data["Jet_phi_bnd_log"],
        data["Jet_mass_bnd_log"],
        data["Jet_eta_brd"],
        data["Jet_pt_brd"],
        data["Jet_phi_brd"],
        data["Jet_mass_brd"],
        data["Jet_eta_brd_log"],
        data["Jet_pt_brd_log"],
        data["Jet_phi_brd_log"],
        data["Jet_mass_brd_log"],
        data["MET_covXX"],
        data["MET_covXY"],
        data["MET_covYY"],
        data["MET_phi"],
        data["MET_pt"],
        data["MET_significance"],
        data["MET_pt_log"],
        data["m_hh"]
    ]).T

    npTarget = data["GenMET_pt"]

    print(f'(N events, N features): {npFeatures.shape}')
    return npFeatures, npTarget

"""
    Defines the difference between true MET and measured MET as
    the correction for this regression model
"""
def MET_correction(data):
    METcorr = data["MET_pt"] - data["GenMET_pt"]

    print(f'✅ Calculated MET correction for the regression model\nSome info on MET correction:')
    print(f'METcorr MIN: {np.min(METcorr)}')
    print(f'METcorr MAX: {np.max(METcorr)}')
    print(f'METcorr MEAN: {np.mean(METcorr)}')
    print(f'METcorr MEDIAN: {np.median(METcorr)}')
    return METcorr

"""
    Training XGBoost model using all features
"""
def model_training(npFeatures, METcorr):
    train_start_time = time.time()

    # Splitting dataset in training and testing
    x_train, x_test, y_train, y_test = train_test_split(npFeatures, METcorr, test_size=0.2, random_state=42)

    x_train_red, x_val, y_train_red, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    # Defining KFold
    kf = KFold(
        n_splits = 3,
        shuffle = True,
        random_state = 42
    )

    # Model definition
    model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42
    )

    # Hyper-parameters searching grid
    hparam_grid = {
        'n_estimators': [600, 800, 1000],  # 600
        'learning_rate': [0.01, 0.05],   #0.05
        'max_depth': [4, 6],   # 6
        'min_child_weight': [1],  #1
        'subsample': [0.8, 1],  #0.8
        'colsample_bytree': [1],  #1
        'reg_alpha': [1],
        'reg_lambda': [1]
    }

    # Grid search with cross validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=hparam_grid,
        scoring='neg_root_mean_squared_error',
        cv=kf,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )

    # Fits with combinations of hparams
    grid_search.fit(
        x_train, y_train
    )

    print(f"Best parameters found:\n {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test)

    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    print(f'✅Training completed in {train_time}s')

    # Saving model
    best_model.save_model('bestmodel_allfeatures.json')
    
    print(f'✅Saved best model, all features included')
    return grid_search, best_model, y_test, y_pred

"""
    Evaluating the regression model using different metrics
"""
def evaluate_regression(y_test, y_pred, grid_search):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Evaluation metrics for the model:')
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    
    results = grid_search.cv_results_

    for i in range(len(results["params"])):
        params = results["params"][i]
        train_score = -results["mean_train_score"][i]  # Negated back to RMSE
        val_score = -results["mean_test_score"][i]     # Negated back to RMSE

        print(f"Params: {params}")
        print(f"  Training RMSE: {train_score:.4f}")
        print(f"  Validation RMSE: {val_score:.4f}")
        print("-" * 40)
    return rmse, mae, r2

def select_top_features_cumulative(params, npFeatures, best_model):
    # Getting sorted features importance dictionary
    importances = best_model.feature_importances_
    feature_importance_dict = {
        feature: importance
        for feature, importance in sorted(zip(params, importances),
        key=lambda x: x[1],
        reverse=True)
    }

    print(f'Features importance dictionary:\n {feature_importance_dict}')

    # Select best features
    sorted_idx = np.argsort(importances)[::-1]
    #cumulative_importance = np.cumsum(importances[sorted_idx])
    selected_idx = sorted_idx[:10]

    # Print selected feature names
    selected_feature_names = [params[i] for i in selected_idx]
    print(f'Selected features ({len(selected_feature_names)}): {selected_feature_names}')

    features_selected = npFeatures[:, selected_idx]

    return features_selected

"""
    Training XGBoost model using only selected features
"""
def model_training_sel_features(features_selected, METcorr):
    print(f'✅Training new model with chosen features')
    st = time.time()

    # Splitting dataset in training and testing
    x_train, x_test, y_train, y_test = train_test_split(features_selected, METcorr, test_size=0.2, random_state=42)
    
    # Defining KFold
    kf = KFold(
        n_splits = 3,
        shuffle = True,
        random_state = 42
    )

    # Model definition
    model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)
    
    # Hyper-parameters searching grid
    hparam_grid = {
        'n_estimators': [250, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 5],
        'min_child_weight': [3, 5, 7],
        'subsample': [1],
        'colsample_bytree': [0.7, 0.8, 1]
    }

    # Grid search with cross validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=hparam_grid,
        scoring='neg_root_mean_squared_error',
        #scoring = 'neg_mean_absolute_error',
        #scoring = 'r2',
        cv=kf,
        verbose=1,
        n_jobs=-1)
    
    # Fits with combinations of hparams
    grid_search.fit(x_train, y_train)

    print(f"Best parameters found:\n {grid_search.best_params_}")

    best_model_sel_features = grid_search.best_estimator_
    y_pred = best_model_sel_features.predict(x_test)

    et = time.time()
    train_time = et - st

    print(f'✅Training with selected features completed in {train_time}s')
    
    # Saving model
    best_model_sel_features.save_model('bestmodel_selectedfeatures.json')
    
    print(f'✅Saved best model, only selected features included')

    # Evaluating model
    rmse_selfeat, mae_selfeat, r2_selfeat = evaluate_regression(y_test, y_pred)
    return best_model_sel_features

def apply_correction(features_selected, best_model_sel_features, data):
    correctedMET = data['MET_pt'] - best_model_sel_features.predict(features_selected)
    return correctedMET

def MET_resolution(data, correctedMET):
    METres = np.std(data['MET_pt'] - correctedMET)
    print(f"MET RESOLUTION: {METres}")
    return METres
"""
def saving_results():
    # Example data to save
    results = {
        "best_params": best_model.get_params(),
        "test_rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
        "num_test_samples": len(y_test),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to JSON file
    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return
"""



if __name__ == '__main__':
    params, target, data = read_data()

    npFeatures, npTarget = data_engineering(data)
        
    METcorr = MET_correction(data)

    grid_search, best_model, y_test, y_pred = model_training(npFeatures, METcorr)
    
    rmse, mae, r2 = evaluate_regression(y_test, y_pred, grid_search)
    
    features_selected = select_top_features_cumulative(params, npFeatures, best_model)
    
    #best_model_sel_features = model_training_sel_features(features_selected, METcorr)
#
    #correctedMET = apply_correction(features_selected, best_model_sel_features, data)
    #
    #METres = MET_resolution(data, correctedMET)


