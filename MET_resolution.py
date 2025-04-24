# XGBoost has to be imported before ROOT to avoid crashes because of clashing
import xgboost as xgb
#from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import uproot
import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objs as go

"""
    Reads data from ROOT file and selects ML alorithm's parameters.
    To work with training and testing sets, creates a matrix (4000, 7)
    containing the parameters corrisponding to each event.
"""
def read_data():
    file = uproot.open("datasets/skimmed.root")
    tree = file["Events"]

    params = ['nJet', 'Jet_eta', 'Jet_pt', 'Jet_phi', 'Jet_mass',
                'MET_pt', 'MET_significance']
    target = ['GenMET_pt']

    data = tree.arrays(params + target, library="np")

    # Choosing nJet<=3
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

        # Second best Jet params
        Jet_eta_bnd.append(jets_eta[1] if len(jets_eta) > 1 else 0.0)
        Jet_pt_bnd.append(jets_pt[1] if len(jets_pt) > 1 else 0.0)
        Jet_phi_bnd.append(jets_phi[1] if len(jets_phi) > 1 else 0.0)
        Jet_mass_bnd.append(jets_mass[1] if len(jets_mass) > 1 else 0.0)

        # Third best Jet params
        Jet_eta_brd.append(jets_eta[2] if len(jets_eta) > 2 else 0.0)
        Jet_pt_brd.append(jets_pt[2] if len(jets_pt) > 2 else 0.0)
        Jet_phi_brd.append(jets_phi[2] if len(jets_phi) > 2 else 0.0)
        Jet_mass_brd.append(jets_mass[2] if len(jets_mass) > 2 else 0.0)

        # Combined parameters
        m_hh.append(Jet_mass_bst[i] + Jet_mass_bnd[i])
    

    npFeatures = np.array([
        #data["nJet"],   
        Jet_pt_bst,     # in the paper
        Jet_eta_bst,    # in the paper
        #Jet_phi_bst,
        Jet_mass_bst,
        Jet_pt_bnd,     # in the paper
        Jet_eta_bnd,    # in the paper
        #Jet_phi_bnd,
        Jet_mass_bnd,
        #m_hh,
        data["MET_pt"],
        data["MET_significance"]   #in the paper
    ]).T
    
    print(npFeatures.shape)
    MET = data["MET_pt"]
    print(f'MET: {MET}')

    npTarget = data["GenMET_pt"]
    
    return npFeatures, npTarget, data

"""
    Defines the difference between true MET and measured MET as
    the correction for this regression model.
"""
def MET_correction(data):
    METcorr = data["MET_pt"] - data["GenMET_pt"]
    return METcorr

"""
    Train XGBoost model
"""
def model_training(npFeatures, METcorr):
    x_train, x_test, y_train, y_test = train_test_split(npFeatures, METcorr, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    dtrain = xgb.DMatrix(x_train_scaled, label=y_train)
    dtest = xgb.DMatrix(x_test_scaled, label=y_test)

    # Choosing hyper-parameters
    hparams = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    }

    num_rounds = 100
    model = xgb.train(hparams, dtrain, num_rounds, evals=[(dtest, 'test')], early_stopping_rounds=10)

    target_prediction = model.predict(dtest)
    mse = mean_squared_error(y_test, target_prediction)
    return mse, model

def apply_correction(npFeatures, model, data):
    features_dmatrix = xgb.DMatrix(npFeatures)
    correctedMET = data['MET_pt'] - model.predict(features_dmatrix)
    mse_after_correction = mean_squared_error(data["GenMET_pt"], correctedMET)
    print(f"REAL RMSE = {np.sqrt(mse_after_correction)}")
    return correctedMET

def MET_resolution(data, correctedMET):
    METres = np.std(data['MET_pt'] - correctedMET)
    print(f"MET RESOLUTION: {METres}")
    return METres

"""
    Dashboard
"""
def dashboard(npTarget, correctedMET):
    app = dash.Dash(__name__)
    app.title = "MET Resolution Dashboard"

    dark_background = '#1e1e1e'
    light_text = '#ffffff'

    app.layout = html.Div(
        style={'backgroundColor': dark_background, 'minHeight': '100vh', 'padding': '20px'},
        children=[
            html.H1(
                "MET Resolution Model Evaluation",
                style={'textAlign':'center', 'color':light_text}
            ),

        html.Div([
            html.P(
                "True MET VS Corrected MET",
                style={'fontSize': 18, 'color':light_text}
            )
        ], style={'textAlign':'left'}),

        dcc.Graph(
            id='scatter-plot',
            figure={
                'data': [
                    go.Scatter(x=npTarget, y=correctedMET, mode='markers',
                            marker=dict(color='lightblue', size=6),
                            name='Predicted vs True')
                ],
                'layout': go.Layout(
                    title='True MET VS Corrected',
                    paper_bgcolor=dark_background,
                    plot_bgcolor=dark_background,
                    font=dict(color=light_text),
                    xaxis=dict(title='True MET', color=light_text, gridcolor='#444'),
                    yaxis=dict(title='Corrected MET', color=light_text, gridcolor='#444'),
                    hovermode='closest'
                )
            }
        )
    ])
    return app


if __name__ == '__main__':
    npFeatures, npTarget, data = read_data()
        
    METcorr = MET_correction(data)
    mse, model = model_training(npFeatures, METcorr)
    correctedMET = apply_correction(npFeatures, model, data)
    METres = MET_resolution(data, correctedMET)
    app = dashboard(npTarget, correctedMET)

    raw_met = data['MET_pt']

    app.run_server(debug=True)


