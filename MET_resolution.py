# XGBoost has to be imported before ROOT to avoid crashes because of clashing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import uproot

"""
    Reads data from ROOT file and selects ML alorithm's parameters.
    To work with training and testing sets, creates a matrix (4000, 7)
    containing the parameters corrisponding to each event.
"""
def read_data():
    file = uproot.open("skimmed.root")
    tree = file["Events"]

    features = ['nJet', 'Jet_eta', 'Jet_mass', 'Jet_phi', 'Jet_pt',
                'MET_pt', 'MET_significance']
    target = ['GenMET_pt']

    data = tree.arrays(features + target, library="np")

    npFeatures=[]
    n_events = len(data[features[0]])
    for i in range(n_events):
        row = []
        for feat in features:
            val = data[feat][i]
            # Mean of the values inside each feature's array per event
            if isinstance(val, np.ndarray):
                # Arrays with less jets are shorter -> replace NaNs with zeros 
                if val.size > 0:
                    row.append(np.mean(val))
                else:
                    row.append(0.0)
            else:
                row.append(val)
        npFeatures.append(row)

    npTarget = data["GenMET_pt"]

    #for key in features + target:
    #    print(f"{key}: {data[key].shape}")
    #print(f'{npFeatures.shape}')
    #print(npFeatures.dtype)  
    
    return npFeatures, npTarget, data

"""
    Defines the difference between true MET and measured MET as
    the correction for this regression model.
"""
def MET_correction(data):
    METcorr = data['GenMET_pt'] - data['MET_pt']
    print(f"MET CORRECTION: {METcorr}")
    return METcorr

"""
    Train XGBoost model
"""
def model_training(npFeatures, METcorr):
    x_train, x_test, y_train, y_test = train_test_split(npFeatures, METcorr, test_size=0.2, random_state=42)
    model = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
    model.fit(x_train, y_train)
    target_prediction = model.predict(x_test)
    mse = mean_squared_error(y_test, target_prediction)
    return mse, model

def apply_correction(npFeatures, model, data):    
    correctedMET = data['MET_pt'] + model.predict(npFeatures)
    return correctedMET

def MET_resolution(data, correctedMET):
    METres = np.std(data['MET_pt'] - correctedMET)
    return METres

if __name__ == '__main__':
    npFeatures, npTarget, data = read_data()
        
    METcorr = MET_correction(data)
    mse, model = model_training(npFeatures, METcorr)
    correctedMET = apply_correction(npFeatures, model, data)
    METres = MET_resolution(data, correctedMET)

    raw_met = data['MET_pt']

    for i in range(len(raw_met)):
        print(f'Iteration {i}\n---------------')
        print(f'Raw MET: {raw_met[i]}')
        print(f'True MET: {npTarget[i]}')
        print(f'Corrected MET: {correctedMET[i]}')
        print(f'MET resolution: {METres}\n')

