# XGBoost has to be imported before ROOT to avoid crashes because of clashing
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import uproot

"""
    Returns the log value of the elements inside each array
"""
def log_params(array):
    log_array = np.log(np.array(array) + 6)
    return log_array

"""
    Reads data from ROOT file and selects ML alorithm's parameters
"""
def read_data():
    file = uproot.open("datasets/skimmed1.root")
    tree = file["Events"]

    params = [
        'nJet', 'Jet_eta', 'Jet_pt', 'Jet_phi', 'Jet_mass',
        'MET_pt', 'MET_phi',
        'MET_covXX', 'MET_covXY', 'MET_covYY', 'MET_significance'
    ]
    target = ['GenMET_pt']

    data = tree.arrays(params + target, library="np")

    print(f'DATA SHAPE: {data["GenMET_pt"].shape}')
    return params, target, data

def feature_retrieving(params, data):
    # Selecting only the first three Jets
    Jet_eta_bst = []
    Jet_pt_bst = []
    Jet_phi_bst = []
    Jet_mass_bst = []
    Jet_eta_bnd = []
    Jet_pt_bnd = []
    Jet_phi_bnd = []
    Jet_mass_bnd = []
    Jet_eta_brd = []
    Jet_pt_brd = []
    Jet_phi_brd = []
    Jet_mass_brd = []

    m_hh = []
    
    n_events = len(data[params[0]])
    for i in range(n_events):
        jets_eta = data["Jet_eta"][i]
        jets_pt = data["Jet_pt"][i]
        jets_phi = data["Jet_phi"][i]
        jets_mass = data["Jet_mass"][i]

        # Best Jet params
        Jet_eta_bst.append(jets_eta[0] if len(jets_eta) > 0 else 0.0)
        Jet_pt_bst.append(jets_pt[0] if len(jets_pt) > 0 else 0.0)
        Jet_phi_bst.append(jets_phi[0] if len(jets_phi) > 0 else 0.0)
        Jet_mass_bst.append(jets_mass[0] if len(jets_mass) > 0 else 0.0)

        #Jet_eta_bst_log = log_params(Jet_eta_bst)
        #Jet_pt_bst_log = log_params(Jet_pt_bst)
        #Jet_phi_bst_log = log_params(Jet_phi_bst)
        #Jet_mass_bst_log = log_params(Jet_mass_bst)

        # Second best Jet params
        Jet_eta_bnd.append(jets_eta[1] if len(jets_eta) > 1 else 0.0)
        Jet_pt_bnd.append(jets_pt[1] if len(jets_pt) > 1 else 0.0)
        Jet_phi_bnd.append(jets_phi[1] if len(jets_phi) > 1 else 0.0)
        Jet_mass_bnd.append(jets_mass[1] if len(jets_mass) > 1 else 0.0)

        #Jet_eta_bnd_log = log_params(Jet_eta_bnd)
        #Jet_pt_bnd_log = log_params(Jet_pt_bnd)
        #Jet_phi_bnd_log = log_params(Jet_phi_bnd)
        #Jet_mass_bnd_log = log_params(Jet_mass_bnd)

        # Third best Jet params
        Jet_eta_brd.append(jets_eta[2] if len(jets_eta) > 2 else 0.0)
        Jet_pt_brd.append(jets_pt[2] if len(jets_pt) > 2 else 0.0)
        Jet_phi_brd.append(jets_phi[2] if len(jets_phi) > 2 else 0.0)
        Jet_mass_brd.append(jets_mass[2] if len(jets_mass) > 2 else 0.0)

        #Jet_eta_brd_log = log_params(Jet_eta_brd)
        #Jet_pt_brd_log = log_params(Jet_pt_brd)
        #Jet_phi_brd_log = log_params(Jet_phi_brd)
        #Jet_mass_brd_log = log_params(Jet_mass_brd)

        # Combined parameters
        m_hh.append(Jet_mass_bst[i] + Jet_mass_bnd[i])

    print(np.min(Jet_eta_bst))
    print(np.min(Jet_pt_bst))
    print(np.min(Jet_phi_bst))
    print(np.min(Jet_mass_bst))

    MET_pt = data["MET_pt"]
    MET_pt_log = log_params(MET_pt)

    npFeatures = np.array([   
        Jet_pt_bst,     # in the paper
        Jet_eta_bst,    # in the paper
        Jet_phi_bst,
        Jet_mass_bst,
        #Jet_pt_bst_log,
        #Jet_eta_bst_log,
        #Jet_phi_bst_log,
        #Jet_mass_bst_log,
        Jet_pt_bnd,     # in the paper
        Jet_eta_bnd,    # in the paper
        Jet_phi_bnd,
        Jet_mass_bnd,
        #Jet_pt_bnd_log,     # in the paper
        #Jet_eta_bnd_log,    # in the paper
        #Jet_phi_bnd_log,
        #Jet_mass_bnd_log,
        Jet_pt_brd,
        Jet_eta_brd,
        Jet_phi_brd,
        Jet_mass_brd,
        #Jet_pt_brd_log,
        #Jet_eta_brd_log,
        #Jet_phi_brd_log,
        #Jet_mass_brd_log,
        m_hh,
        MET_pt,
        MET_pt_log,
        data["MET_phi"],
        data["MET_covXX"],
        data["MET_covXY"],
        data["MET_covYY"],
        data["MET_significance"]   #in the paper
    ]).T

    print(f'(N events, N features): {npFeatures.shape}')
    npTarget = data["GenMET_pt"]
    
    return npFeatures, npTarget

"""
    Defines the difference between true MET and measured MET as
    the correction for this regression model
"""
def MET_correction(data):
    METcorr = data["MET_pt"] - data["GenMET_pt"]

    print(f'METcorr MIN: {np.min(METcorr)}')
    print(f'METcorr MAX: {np.max(METcorr)}')
    print(f'METcorr MEAN: {np.mean(METcorr)}')
    print(f'METcorr MEDIAN: {np.median(METcorr)}')
    return METcorr

"""
    XGBoost model
"""
def model_training(npFeatures, METcorr):
    # Splitting dataset in training and testing
    x_train, x_test, y_train, y_test = train_test_split(npFeatures, METcorr, test_size=0.2, random_state=42)

    # Defining KFold
    kf = KFold(
        n_splits = 3,
        shuffle = True,
        random_state = 42
    )

    # Model definition
    model = XGBRegressor(objective='reg:squarederror', random_state=42)

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
        #scoring='neg_root_mean_squared_error',
        #scoring = 'neg_mean_absolute_error',
        scoring = 'r2',
        cv=kf,
        verbose=1,
        n_jobs=-1)

    grid_search.fit(x_train, y_train)

    print(f"Best parameters found:\n {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test)
    return best_model, y_test, y_pred

def evaluate_regression(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    return

def select_top_features_cumulative(npFeatures, best_model, cumulative_threshold=0.80):
    feature_names = [
        "Jet_pt_bst",
        "Jet_eta_bst",
        "Jet_phi_bst",
        "Jet_mass_bst",
        #"Jet_pt_bst_log",
        #"Jet_eta_bst_log",
        #"Jet_phi_bst_log",
        #"Jet_mass_bst_log",
        "Jet_pt_bnd",
        "Jet_eta_bnd",
        "Jet_phi_bnd",
        "Jet_mass_bnd",
        #"Jet_pt_bnd_log",
        #"Jet_eta_bnd_log",
        #"Jet_phi_bnd_log",
        #"Jet_mass_bnd_log",
        "Jet_pt_brd",
        "Jet_eta_brd",
        "Jet_phi_brd",
        "Jet_mass_brd",
        #"Jet_pt_brd_log",
        #"Jet_eta_brd_log",
        #"Jet_phi_brd_log",
        #"Jet_mass_brd_log",
        "m_hh",
        "MET_pt",
        "MET_pt_log",
        "MET_phi",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        "MET_significance"
    ]
    importances = best_model.feature_importances_
    feature_importance_dict = {
        feature: importance
        for feature, importance in sorted(zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True)
    }
    print(f'FEATURES_IMPORTANCE: {feature_importance_dict}')

    sorted_idx = np.argsort(importances)[::-1]
    cumulative_importance = np.cumsum(importances[sorted_idx])
    selected_idx = sorted_idx[np.where(cumulative_importance <= cumulative_threshold)[0]]

    if len(selected_idx) == 0:
        selected_idx = sorted_idx[:1]

    features_selected = npFeatures[:, selected_idx]

    return features_selected, selected_idx

def apply_correction(npFeatures, model, data):
    features_dmatrix = xgb.DMatrix(npFeatures)
    correctedMET = data['MET_pt'] - model.predict(features_dmatrix)
    #mse_after_correction = mean_squared_error(data["GenMET_pt"], correctedMET)
    #print(f"REAL RMSE = {np.sqrt(mse_after_correction)}")
    return correctedMET

def MET_resolution(data, correctedMET):
    METres = np.std(data['MET_pt'] - correctedMET)
    print(f"MET RESOLUTION: {METres}")
    return METres



#import dash
#from dash import Dash, html, dcc, callback, Output, Input
#import plotly.graph_objs as go
#import dash_bootstrap_components as dbc
"""
    Dashboard
"""
def dashboard(npTarget, correctedMET):
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.theme.DARKLY]
    )
    app.title = "MET Resolution Dashboard"

    dark_background = '#1e1e1e'
    light_text = '#ffffff'

    app.layout = dbc.Container([
        html.H1(
            'MET Resolution Model Evaluation',
            style={'textAlign': 'center', 'color': 'white'}),

        html.Div([
            html.P(
                "True MET VS Corrected MET",
                style={'fontSize': 18, 'color': 'white'}
            )
        ], style={'textAlign': 'left'}),

        dcc.Graph(
            id='scatter-plot',
            figure={
                'data': [
                    go.Scatter(
                        x=npTarget, y=correctedMET, mode='markers',
                        marker=dict(color='lightblue', size=6),
                        name='Predicted vs True'
                    )
                ],
                'layout': go.Layout(
                    title='True MET VS Corrected',
                    paper_bgcolor='#2c3e50',  # dark background
                    plot_bgcolor='#2c3e50',
                    font=dict(color='white'),
                    xaxis=dict(title='True MET', color='white'),
                    yaxis=dict(title='Corrected MET', color='white'),
                    hovermode='closest'
                )
            }
        )
    ], fluid=True)
    return app


if __name__ == '__main__':
    params, target, data = read_data()
    npFeatures, npTarget = feature_retrieving(params, data)
        
    METcorr = MET_correction(data)
    best_model, y_test, y_pred = model_training(npFeatures, METcorr)
    evaluate_regression(y_test, y_pred)
    features_selected, selected_idx = select_top_features_cumulative(npFeatures, best_model, cumulative_threshold=0.80)
    #correctedMET = apply_correction(npFeatures, model, data)
    #METres = MET_resolution(data, correctedMET)
    #app = dashboard(npTarget, correctedMET)
    

    #app.run_server(debug=True)


