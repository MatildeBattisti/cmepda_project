# XGBoost has to be imported before ROOT to avoid crashes because of clashing
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from itertools import product
import uproot
import time
import json

"""
    Reads data from ROOT file and selects ML alorithm's parameters
"""
def read_data():
    file = uproot.open("datasets/skimmed0.root")
    tree = file["Events"]

    params = [
        "nJet",
        "Jet_eta_bst",
        "Jet_pt_bst",
        "Jet_phi_bst",
        "Jet_mass_bst",
        #"Jet_eta_bst_log",
        #"Jet_pt_bst_log",
        #"Jet_phi_bst_log",
        #"Jet_mass_bst_log",
        "Jet_eta_bnd",
        "Jet_pt_bnd",
        "Jet_phi_bnd",
        "Jet_mass_bnd",
        #"Jet_eta_bnd_log",
        #"Jet_pt_bnd_log",
        #"Jet_phi_bnd_log",
        #"Jet_mass_bnd_log",
        "Jet_eta_brd",
        "Jet_pt_brd",
        "Jet_phi_brd",
        "Jet_mass_brd",
        #"Jet_eta_brd_log",
        #"Jet_pt_brd_log",
        #"Jet_phi_brd_log",
        #"Jet_mass_brd_log",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        "MET_phi",
        "MET_pt",
        "MET_significance",
        #"MET_pt_log",
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
        #data["Jet_eta_bst_log"],
        #data["Jet_pt_bst_log"],
        #data["Jet_phi_bst_log"],
        #data["Jet_mass_bst_log"],
        data["Jet_eta_bnd"],
        data["Jet_pt_bnd"],
        data["Jet_phi_bnd"],
        data["Jet_mass_bnd"],
        #data["Jet_eta_bnd_log"],
        #data["Jet_pt_bnd_log"],
        #data["Jet_phi_bnd_log"],
        #data["Jet_mass_bnd_log"],
        data["Jet_eta_brd"],
        data["Jet_pt_brd"],
        data["Jet_phi_brd"],
        data["Jet_mass_brd"],
        #data["Jet_eta_brd_log"],
        #data["Jet_pt_brd_log"],
        #data["Jet_phi_brd_log"],
        #data["Jet_mass_brd_log"],
        data["MET_covXX"],
        data["MET_covXY"],
        data["MET_covYY"],
        data["MET_phi"],
        data["MET_pt"],
        data["MET_significance"],
        #data["MET_pt_log"],
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
    x_train_full, x_test, y_train_full, y_test = train_test_split(npFeatures, METcorr, test_size=0.2, random_state=42)
    
    # Hyper-parameters searching grid
    hparam_grid = {
        'n_estimators': [400, 600, 800],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'min_child_weight': [1],
        'subsample': [0.8, 1],
        'colsample_bytree': [1],
        'reg_alpha': [1],
        'reg_lambda': [1]
    }

    # Defining KFold
    kf = KFold(
        n_splits = 3,
        shuffle = True,
        random_state = 42
    )

    # Create all combinations of parameters
    param_combinations = list(product(*hparam_grid.values()))
    param_keys = list(hparam_grid.keys())

    best_score = float('inf')
    best_params = None

    # Iterate over each parameter combination
    for combo in param_combinations:
        params = dict(zip(param_keys, combo))
        print(f"🔍 Testing params: {params}")

        val_scores = []

        # K-Fold loop
        for train_idx, val_idx in kf.split(x_train_full):
            x_train, x_val = x_train_full[train_idx], x_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

            model = XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                early_stopping_rounds=20,
                random_state=42,
                **params
            )

            model.fit(
                x_train, y_train,
                eval_set=[(x_val, y_val)],
                verbose=False
            )

            y_pred = model.predict(x_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_scores.append(rmse)

        avg_rmse = np.mean(val_scores)
        print(f"    🔹 Avg RMSE: {avg_rmse:.4f}\n")

        # Save best params
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params

    print("✅ Best Parameters Found:")
    print(best_params)
    print(f"📉 Best Avg RMSE: {best_score:.4f}")

    # Training model with best hparams on full training set
    best_model = XGBRegressor(
        **best_params,
        eval_metric='rmse',
        objective='reg:squarederror',
        early_stopping_rounds=20
    )

    # Splitting training dataset in a smaller train and validation dataset
    x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

    best_model.fit(
        x_train_final, y_train_final,
        eval_set=[(x_val_final, y_val_final)],
        verbose=True
    )

    y_test_pred = best_model.predict(x_test)

    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    print(f'✅Training completed in {train_time}s')

    # Saving model
    best_model.save_model('utils/bestmodel.json')
    
    print(f'✅Saved best model, all features included')
    return best_model, y_test, y_test_pred

"""
    Evaluating the regression model using different metrics
"""
def evaluate_regression(y_test, y_test_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f'Evaluation metrics for the model:')
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
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
    selected_idx = sorted_idx[:10]

    # Print selected feature names
    selected_feature_names = [params[i] for i in selected_idx]
    print(f'Selected features ({len(selected_feature_names)}): {selected_feature_names}')

    features_selected = npFeatures[:, selected_idx]

    return features_selected

"""
    Adding to the experimental MET the ML predicted correction
"""
def apply_correction(npFeatures, best_model, data):
    correctedMET = data['MET_pt'] - best_model.predict(npFeatures)
    print(f"Corrected MET: {correctedMET}")
    return correctedMET

"""
    Defining the MET resolution as the standard deviation between the
    experimental MET and the ML corrected one
"""
def MET_resolution(data, correctedMET):
    METres = np.std(data['MET_pt'] - correctedMET)
    print(f"MET resolution: {METres}")
    return METres



if __name__ == '__main__':
    params, target, data = read_data()

    npFeatures, npTarget = data_engineering(data)
        
    METcorr = MET_correction(data)

    best_model, y_test, y_test_pred = model_training(npFeatures, METcorr)
    
    rmse, mae, r2 = evaluate_regression(y_test, y_test_pred)
    
    features_selected = select_top_features_cumulative(params, npFeatures, best_model)
    
    correctedMET = apply_correction(npFeatures, best_model, data)
    
    METres = MET_resolution(data, correctedMET)


